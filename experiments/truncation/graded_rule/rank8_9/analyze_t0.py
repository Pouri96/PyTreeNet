"""Merge rank8_9/results/t0_*.json and apply the pre-registered T0 decision table (see results/prereg_interpretations.txt).

    python rank8_9/analyze_t0.py            # writes rank8_9/results/t0_summary.txt
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
LO, HI = 1e-4, 1e-2
NONLOCAL = lambda name: name.startswith('ppp_mo') or name.startswith('lr0.5') or name.startswith('lr1.5')


def load():
    allc = {}
    for f in sorted(glob.glob(str(RES / 't0_[A-Z].json'))):
        allc.update(json.load(open(f)))
    return allc


def usable(rec):
    return [r for r in rec['rows'] if LO <= r['infid'] <= HI]


def main():
    allc = load()
    out = []
    P = out.append
    P('T0 locality budget of the SVD-sweep compression error.  usable = SVD infidelity in [1e-4, 1e-2].')
    P('columns: f_loc = unweighted squared-error share of elements with span<=4; fE = H-coefficient-weighted squared share (primary);')
    P('          sgn / abs = signed / absolute share of the energy error from span<=4 elements; leak = max(Var N, Var Sz)')
    P('')
    P(f"{'cell':<14}{'chi':>4}{'infid':>10}{'dE':>10}{'f_loc':>7}{'fE(2)':>7}{'fE(3)':>7}{'fE(4)':>7}{'fE(6)':>7}{'sgn4':>7}{'abs4':>7}{'leak':>9}  use")
    for name, rec in allc.items():
        for r in rec['rows']:
            u = LO <= r['infid'] <= HI
            P(f"{name:<14}{r['chi']:>4}{r['infid']:>10.2e}{r['dE']:>10.2e}{r['loc4']:>7.3f}{r['locE2']:>7.3f}{r['locE3']:>7.3f}{r['locE4']:>7.3f}"
              f"{r['locE6']:>7.3f}{r['locEs4']:>7.2f}{r['locEa4']:>7.3f}{r.get('leak', float('nan')):>9.1e}  {'*' if u else ''}")
        P('')
    # decision
    P('Decision table (primary reading: f_loc^E = H-weighted squared share, span<=4)')
    for label, key in (('primary fE(4) squared-weighted', 'locE4'), ('alt: signed energy share', 'locEs4'), ('alt: absolute energy share', 'locEa4'),
                       ('alt: fE(3) (spans actually targeted by ks=(1,2,3))', 'locE3'), ('alt: unweighted f_loc(4)', 'loc4')):
        for scope in ('all orderings', 'natural ordering only'):
            vals = []
            for name, rec in allc.items():
                if not NONLOCAL(name):
                    continue
                if scope.startswith('natural') and not name.endswith('_nat'):
                    continue
                for r in usable(rec):
                    vals.append((name, r['chi'], r[key]))
            if not vals:
                P(f'  {label:<55}{scope:<24}: no usable entries')
                continue
            mx = max(vals, key=lambda t: t[2])
            mn = min(vals, key=lambda t: t[2])
            kill = all(v[2] < 0.25 for v in vals)
            proceed = any(v[2] >= 0.4 for v in vals)
            verdict = 'KILL (-> T2)' if kill else ('PROCEED (-> T1)' if proceed else 'AMBIGUOUS (T1 on best ordering)')
            P(f'  {label:<55}{scope:<24}: n={len(vals):<3} min={mn[2]:.3f} ({mn[0]},chi={mn[1]})  max={mx[2]:.3f} ({mx[0]},chi={mx[1]})  -> {verdict}')
    P('')
    P('Per-family ranges of the usable entries (min .. max over usable chi and orderings), by reading:')
    fams = {'MO-basis PPP': lambda n: n.startswith('ppp_mo'), 'long-range Ising alpha=0.5': lambda n: n.startswith('lr0.5'),
            'long-range Ising alpha=1.5': lambda n: n.startswith('lr1.5'), 'site-basis PPP (extra)': lambda n: n.startswith('ppp_site'),
            'long-range Ising alpha=3 (extra)': lambda n: n.startswith('lr3.0')}
    for fam, sel in fams.items():
        for order_sel, lab in ((lambda n: True, 'all orderings'), (lambda n: n.endswith('_nat'), 'natural')):
            row = []
            for key in ('locE4', 'locEa4', 'locEs4', 'loc4'):
                v = [r[key] for n, rec in allc.items() if sel(n) and order_sel(n) for r in usable(rec)]
                row.append(f'{key}: {min(v):.2f}..{max(v):.2f}' if v else f'{key}: -')
            P(f'  {fam:<34}{lab:<14} ' + '   '.join(row))
    P('')
    P('Ordering rescue (best ordering / natural ordering of f_loc^E(4) and f_loc(4), same basis, chi with infidelity in the window for both)')
    groups = {}
    for name in allc:
        base, _, order = name.rpartition('_')
        groups.setdefault(base, {})[order] = allc[name]
    for base, g in groups.items():
        if 'nat' not in g or len(g) < 2:
            continue
        for key in ('locE4', 'loc4'):
            for order in g:
                if order == 'nat':
                    continue
                rows = []
                for rn in usable(g['nat']):
                    for ro in g[order]['rows']:
                        if ro['chi'] == rn['chi'] and LO <= ro['infid'] <= HI:
                            rows.append((rn['chi'], rn[key], ro[key]))
                if rows:
                    P(f'  {base:<10} {order:<5} {key:<6}: ' + '  '.join(f'chi={c}: nat {a:.3f} -> {order} {b:.3f} (x{b / max(a, 1e-12):.2f})' for c, a, b in rows))
    txt = '\n'.join(out)
    (RES / 't0_summary.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
