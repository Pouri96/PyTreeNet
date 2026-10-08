"""Evaluate the pre-registered Test 2 criteria (see test2.py) on results/test2_raw.jsonl."""
import _p6  # noqa: F401
import json
import sys
import numpy as np

R = {}
for l in open('results/test2_raw.jsonl'):
    r = json.loads(l)
    R[r['job']] = r
G = [0.0, 0.01, 0.03, 0.1]
C = [6, 8, 12]
K = 4


def g(arm, gm, chi):
    return R.get(f'{arm}_g{gm:g}_chi{chi}_k{K}')


rows = []
print('gamma chi |  nn_svd   nn_spcf  gain | nnn gain | sgl gain |  E gain | tn_svd  tn_spcf  tn ratio | far3 ratio far4 ratio | fired/calls | gated: gain tn_ratio fired')
for gm in G:
    for chi in C:
        s, p, q = g('svd', gm, chi), g('spcf', gm, chi), g('gated', gm, chi)
        if not (s and p):
            continue
        row = dict(gamma=gm, chi=chi, nn_svd=s['nn_rms'], nn_spcf=p['nn_rms'], gain=s['nn_rms'] / p['nn_rms'], gain_nnn=s['nnn_rms'] / p['nnn_rms'],
                   gain_single=s['single_rms'] / p['single_rms'], gain_E=s['E_abs'] / max(p['E_abs'], 1e-300),
                   tn_svd=s['trace_norm'], tn_spcf=p['trace_norm'], tn_ratio=p['trace_norm'] / s['trace_norm'],
                   far3_ratio=p['far3_rms'] / s['far3_rms'], far4_ratio=p['far4_rms'] / s['far4_rms'], fired=p['fired'], calls=p['calls'],
                   hs_ratio=p['hs_norm'] / s['hs_norm'])
        if q:
            row.update(g_gain=s['nn_rms'] / q['nn_rms'], g_tn_ratio=q['trace_norm'] / s['trace_norm'], g_fired=q['fired'], g_calls=q['calls'])
        rows.append(row)
        print(f"{gm:<5g} {chi:3d} | {row['nn_svd']:.2e} {row['nn_spcf']:.2e} {row['gain']:5.2f} | {row['gain_nnn']:5.2f} | {row['gain_single']:5.2f} | {row['gain_E']:6.2f} | {row['tn_svd']:.3f} {row['tn_spcf']:.3f} {row['tn_ratio']:.3f} | {row['far3_ratio']:.2f} {row['far4_ratio']:.2f} | {row['fired']}/{row['calls']} | "
              + (f"{row['g_gain']:.2f} {row['g_tn_ratio']:.3f} {row['g_fired']}" if q else '-'))
json.dump(rows, open('results/test2_table.json', 'w'), indent=1)

print('\nSUCCESS: gain >= 2 at two adjacent chi in a gamma>=0.01 cell, tn <= 1.10x, far ZZ <= 1.5x;  gamma=0.01 gain >= half the gamma=0 gain')
for gm in G[1:]:
    rr = [r for r in rows if r['gamma'] == gm]
    ok = [r['chi'] for r in rr if r['gain'] >= 2 and r['tn_ratio'] <= 1.10 and r['far3_ratio'] <= 1.5 and r['far4_ratio'] <= 1.5]
    adj = bool((6 in ok and 8 in ok) or (8 in ok and 12 in ok))
    print(f'  gamma={gm}: chi satisfying gain/tn/far: {ok}; two adjacent chi: {adj}')
g0 = {r['chi']: r['gain'] for r in rows if r['gamma'] == 0.0}
g01 = {r['chi']: r['gain'] for r in rows if r['gamma'] == 0.01}
print('  gamma=0 gains', {k: round(v, 2) for k, v in g0.items()}, ' gamma=0.01 gains', {k: round(v, 2) for k, v in g01.items()},
      ' ratio', {k: round(g01[k] / g0[k], 2) for k in g0 if k in g01})
print('KILL: gain < 1.3 in every gamma>=0.01 cell?', all(r['gain'] < 1.3 for r in rows if r['gamma'] >= 0.01),
      ' | trace norm > 1.25x wherever gain >= 1.3?', [(r['gamma'], r['chi'], round(r['tn_ratio'], 2)) for r in rows if r['gamma'] >= 0.01 and r['gain'] >= 1.3 and r['tn_ratio'] > 1.25])
