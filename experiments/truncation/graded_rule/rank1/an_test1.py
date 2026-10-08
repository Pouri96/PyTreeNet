"""Test 1 tables from rank1/results/test1_{model}.json (svd / svdraw / rw:1.6 on the shared dense reference)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import op_bench as B

T, dt, N = 6.0, 0.1, 10
lines = []


def P(s=''):
    print(s)
    lines.append(s)


def row(m):
    d = m['dC_Z']
    lag = m['lag_Z']
    inf = m.get('final', {}).get('infid', float('nan'))
    return (f"{m['arm']:7s} chi={m['chi']:3d} | dC_Z rms ahead {d['ahead']['rms']:.1e} front {d['front']['rms']:.1e} behind {d['behind']['rms']:.3e} "
            f"(max {d['behind']['max']:.3f}, mean {d['behind']['mean']:+.3f}) | dw rms behind {m['dw']['behind']['rms']:.3e} | "
            f"lag0.1 max {lag['0.1']['max']:.3f} lag0.5 max {lag['0.5']['max']:.3f} (cens {lag['0.5']['censored']}) | infid {inf:.3e} | dp1(T) {m['dp1_rms_T']:.2e}")


summary = {}
for model in sys.argv[1:] or ['ising', 'tfim']:
    recs = json.load(open(HERE / 'results' / f'test1_{model}.json'))
    ref = B.get_ref(model, N, T, dt)
    P(f'=== {model}, N={N}, T={T}, dt={dt} ===')
    Ce = B.O.C_from_p1(ref['p1'][1:])
    P(f'exact C_Z: max over x,t = {Ce.max():.3f}; final-time C_Z(x) = ' + ' '.join(f'{v:.2f}' for v in Ce[-1]))
    ms = []
    for rec in sorted(recs, key=lambda r: (r['arm'], r['chi'])):
        m = B.metrics(rec, ref)
        ms.append(m)
        if rec['arm'] == 'svd':
            ms.append(B.metrics(rec, ref, raw=True))
    for m in sorted(ms, key=lambda m: (m['arm'], m['chi'])):
        P(row(m))
    summary[model] = ms
    P()
    if model == 'ising':
        P('-- ratio of per-cut span-2 residual to final one-site marginal error (svd arm) --')
        for m in sorted([m for m in ms if m['arm'] == 'svd'], key=lambda m: m['chi']):
            P(f"chi={m['chi']:3d}: per-cut span-2 rms (last 5 steps) {m['cut_span2_rms_last5']:.3e}  rms dp1(T) {m['dp1_rms_T']:.3e}  ratio {m['ratio_cut_to_final']:.4f}   [all-cuts span-2 rms {m['cut_span2_rms_all']:.2e}, n cuts {m['n_trunc_cuts']}]")
        P('-- svdraw: cumulative kept weight (norm^2 of the unrenormalised operator) at T --')
        for m in sorted([m for m in ms if m['arm'] == 'svdraw'], key=lambda m: m['chi']):
            P(f"chi={m['chi']:3d}: norm^2(T) = {m['norm2_T']:.4f}")
        P('-- rw:1.6 minus svd on C_Z behind the front (positive = rw worse) --')
        for chi in sorted({m['chi'] for m in ms}):
            a = [m for m in ms if m['arm'] == 'svd' and m['chi'] == chi]
            b = [m for m in ms if m['arm'] == 'rw:1.6' and m['chi'] == chi]
            if a and b:
                P(f"chi={chi:3d}: behind rms svd {a[0]['dC_Z']['behind']['rms']:.3e} rw {b[0]['dC_Z']['behind']['rms']:.3e}  ratio rw/svd {b[0]['dC_Z']['behind']['rms'] / a[0]['dC_Z']['behind']['rms']:.2f};  "
                  f"front rms svd {a[0]['dC_Z']['front']['rms']:.3e} rw {b[0]['dC_Z']['front']['rms']:.3e}")
        P()
open(HERE / 'results' / 'test1_summary.txt', 'w').write('\n'.join(lines) + '\n')
