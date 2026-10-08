"""Test 2 tables and verdict from rank1/results/test2_ising.json."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np

recs = json.load(open(HERE / 'results' / 'test2_ising.json'))
lines = []


def P(s=''):
    print(s)
    lines.append(s)


def key(r):
    return (r['step'], r['chi'], r['arm'])


R = {key(r): r for r in recs}
steps = sorted({r['step'] for r in recs})
chis = sorted({r['chi'] for r in recs})
arms = ['svd', 'spcop', 'spcop_k1', 'spcop_k12', 'spcop_p3']
METS = [('win1_rms', '1-site'), ('win2_rms', '2-site'), ('win3_rms', '3-site'), ('far_ge3_rms', 'far d>=3'), ('ZZ_otoc_d3_rms', 'ZZ-OTOC d3'),
        ('infid', 'infid'), ('dp1_rms', 'p1 (=C)')]
P('Test 2: static one-sweep compression of the exact |O(t)>> (Ising, N=10). Ratios are arm / svd at equal chi.')
for st in steps:
    P(f'\n--- t = {st * 0.1:.1f} {"(primary)" if st in (20, 40) else "(supplementary)"} ---')
    for chi in chis:
        base = R.get((st, chi, 'svd'))
        if base is None:
            continue
        P(f"chi={chi}: svd absolute: " + ' '.join(f"{n} {base[k]:.2e}" for k, n in METS) +
          f" | dC behind rms {base['dC']['behind']['rms']:.2e} (n={base['dC']['behind']['n']}) front {base['dC']['front']['rms']:.2e} (n={base['dC']['front']['n']})")
        for a in arms[1:]:
            r = R.get((st, chi, a))
            if r is None:
                continue
            rat = ' '.join(f"{n} {r[k] / base[k]:.2f}" for k, n in METS)
            cb = r['dC']['behind']['rms'] / base['dC']['behind']['rms'] if base['dC']['behind']['rms'] > 0 else float('nan')
            cf = r['dC']['front']['rms'] / base['dC']['front']['rms'] if base['dC']['front']['rms'] > 0 else float('nan')
            P(f"   {a:10s} ratio: {rat} | C behind {cb:.2f} front {cf:.2f} | fired {r.get('fired', '-')}/{r.get('trunc', '-')} wall {r['wall']:.1f}s")

# ---- pre-registered reading
P('\n=== pre-registered reading (spcop vs svd) ===')
P('Continue: ratio on 1-site AND 2-site marginals <= 0.5 at chi = 8 or 16, infidelity ratio <= 1.2, far-marginal (d>=3) ratio <= 1.5.')
P('Kill: ratio >= 0.8 at all chi.   Ambiguous: 0.5 - 0.8.')
for st in (20, 40):
    P(f'-- t = {st * 0.1:.1f} --')
    allr = []
    for chi in chis:
        b, r = R[(st, chi, 'svd')], R[(st, chi, 'spcop')]
        r1, r2, r3 = r['win1_rms'] / b['win1_rms'], r['win2_rms'] / b['win2_rms'], r['win3_rms'] / b['win3_rms']
        ri, rf = r['infid'] / b['infid'], r['far_ge3_rms'] / b['far_ge3_rms']
        allr.append(max(r1, r2))
        P(f'chi={chi:2d}: r1={r1:.2f} r2={r2:.2f} r3={r3:.2f} infid={ri:.2f} far={rf:.2f}   -> worst-of(r1,r2)={max(r1, r2):.2f}; '
          f'meets continue-criteria: {max(r1, r2) <= 0.5 and ri <= 1.2 and rf <= 1.5}')
    P(f'   min over chi of worst-of(r1,r2): {min(allr):.2f};  all chi >= 0.8: {all(a >= 0.8 for a in allr)}')
open(HERE / 'results' / 'test2_summary.txt', 'w').write('\n'.join(lines) + '\n')
