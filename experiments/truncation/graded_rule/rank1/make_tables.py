"""Markdown tables for the results report, from rank1/results/*.json."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import op_bench as B

T, dt, N = 6.0, 0.1, 10
RES = HERE / 'results'


def e(x):
    return f'{x:.2e}'


def load(*fns):
    out = []
    for fn in fns:
        p = RES / fn
        if p.exists():
            out += json.load(open(p))
    return out


def test1():
    ref = B.get_ref('ising', N, T, dt)
    recs = load('test1_ising.json')
    ms = {}
    for r in recs:
        m = B.metrics(r, ref)
        ms[(m['arm'], m['chi'])] = m
        if r['arm'] == 'svd':
            mr = B.metrics(r, ref, raw=True)
            ms[('svdraw', r['chi'])] = mr
    print('### Test 1, Ising (repo model), N=10, T=6, dt=0.1, O = Z_0\n')
    print('| arm | chi | rms dC_Z ahead | front | behind | max behind | mean behind (signed) | rms dw behind | lag th=0.1 (max over sites) | lag th=0.5 (max) | final infid | rms dp1(T) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in ('svd', 'svdraw', 'rw:1.6'):
        for chi in (4, 8, 16, 32):
            m = ms[(arm, chi)]
            d = m['dC_Z']
            lag = m['lag_Z']
            inf = m.get('final', {}).get('infid', float('nan'))
            cens = f" (censored x{lag['0.5']['censored']})" if lag['0.5']['censored'] else ''
            print(f"| {arm} | {chi} | {e(d['ahead']['rms'])} | {e(d['front']['rms'])} | {e(d['behind']['rms'])} | {d['behind']['max']:.3f} | {d['behind']['mean']:+.3f} | {e(m['dw']['behind']['rms'])} | {lag['0.1']['max']:.3f} | {lag['0.5']['max']:.3f}{cens} | {inf:.3f} | {e(m['dp1_rms_T'])} |")
    print()
    print('| chi | norm^2(T) of the unrenormalised operator | svdraw / svd behind-front rms | share of the svdraw rms error removed by renormalising |')
    print('|---|---|---|---|')
    for chi in (4, 8, 16, 32):
        a, b = ms[('svd', chi)], ms[('svdraw', chi)]
        print(f"| {chi} | {b['norm2_T']:.4f} | {b['dC_Z']['behind']['rms'] / a['dC_Z']['behind']['rms']:.2f} | {1 - a['dC_Z']['behind']['rms'] / b['dC_Z']['behind']['rms']:.0%} |")
    print()
    print('| chi | rw:1.6 / svd rms dC_Z behind | rw:1.6 / svd front | per-cut span-2 residual rms (last 5 steps) | rms dp1(T) | ratio |')
    print('|---|---|---|---|---|---|')
    for chi in (4, 8, 16, 32):
        a, b = ms[('svd', chi)], ms[('rw:1.6', chi)]
        print(f"| {chi} | {b['dC_Z']['behind']['rms'] / a['dC_Z']['behind']['rms']:.2f} | {b['dC_Z']['front']['rms'] / a['dC_Z']['front']['rms']:.2f} | {e(a['cut_span2_rms_last5'])} | {e(a['dp1_rms_T'])} | {a['ratio_cut_to_final']:.4f} |")
    print()
    reft = B.get_ref('tfim', N, T, dt)
    recs = load('test1_tfim.json')
    print('### Test 1, TFIM control (hz = 0)\n')
    print('| arm | chi | max abs dC_Z over (x,t) | rms dp1(T) | final infid |')
    print('|---|---|---|---|---|')
    for r in sorted(recs, key=lambda r: (r['arm'], r['chi'])):
        m = B.metrics(r, reft)
        print(f"| {r['arm']} | {r['chi']} | {e(m['dC_Z']['all']['max'])} | {e(m['dp1_rms_T'])} | {m['final']['infid']:.1e} |")
    print()


def test2():
    recs = load('test2_ising.json')
    R = {(r['step'], r['chi'], r['arm']): r for r in recs}
    print('### Test 2: error ratio spcop / svd after ONE left-to-right sweep (Ising N=10, exact |O(t)>>)\n')
    print('| t | chi | svd 1-site rms | spcop/svd 1-site | 2-site | 3-site | far d>=3 | ZZ-OTOC d=3 | infid | C_Z behind (n sites) | C_Z front (n) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for st in (20, 40, 30, 50, 60):
        for chi in (4, 8, 16, 32):
            b, s = R[(st, chi, 'svd')], R[(st, chi, 'spcop')]
            cb = s['dC']['behind']['rms'] / b['dC']['behind']['rms'] if b['dC']['behind']['rms'] > 0 else float('nan')
            cf = s['dC']['front']['rms'] / b['dC']['front']['rms'] if b['dC']['front']['rms'] > 0 else float('nan')
            tag = '' if st in (20, 40) else ' (suppl.)'
            print(f"| {st * dt:.0f}{tag} | {chi} | {e(b['win1_rms'])} | {s['win1_rms'] / b['win1_rms']:.2f} | {s['win2_rms'] / b['win2_rms']:.2f} | {s['win3_rms'] / b['win3_rms']:.2f} | {s['far_ge3_rms'] / b['far_ge3_rms']:.2f} | {s['ZZ_otoc_d3_rms'] / b['ZZ_otoc_d3_rms']:.2f} | {s['infid'] / b['infid']:.2f} | {cb:.2f} ({b['dC']['behind']['n']}) | {cf:.2f} ({b['dC']['front']['n']}) |")
    print()
    print('Other arms at the registered times (ratio to svd; 1-site / 2-site / infid / far):\n')
    print('| t | chi | spcop_k1 (1-site targets only) | spcop_k12 | spcop_p3 (3 Gauss-Newton passes) |')
    print('|---|---|---|---|---|')
    for st in (20, 40):
        for chi in (4, 8, 16):
            b = R[(st, chi, 'svd')]
            cells = []
            for a in ('spcop_k1', 'spcop_k12', 'spcop_p3'):
                s = R[(st, chi, a)]
                cells.append(f"{s['win1_rms'] / b['win1_rms']:.2f} / {s['win2_rms'] / b['win2_rms']:.2f} / {s['infid'] / b['infid']:.2f} / {s['far_ge3_rms'] / b['far_ge3_rms']:.2f}")
            print(f"| {st * dt:.0f} | {chi} | " + ' | '.join(cells) + ' |')
    print()


def test3(model='ising'):
    ref = B.get_ref(model, N, T, dt)
    recs = load(f'test3_{model}.json', f'test3_{model}_gate.json', f'test3_{model}_rw.json', f'test3_{model}_var.json', f'test3_{model}_var32.json', f'test3_{model}_var_k1.json')
    M = {(r['arm'], r['chi']): (B.metrics(r, ref), r) for r in recs}
    chis = sorted({c for (_, c) in M})
    print(f'### Test 3, {model}: ratios to svd (rms of |dC_Z| and |dw| by region), N=10, T=6\n')
    print('| chi | arm | C_Z front | C_Z behind | w front | w behind | lag th=0.5 (svd) | infid | far d>=3 | c_Z0(t) rms | win1 | win2 | win3 | cut CPU vs svd | fired/trunc |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    order = ['svd', 'spcop', 'spcop:gate=1;gate_c0=1.290e-06;gate_exp=0.916', 'spcop:gate=3;gate_c0=1.290e-06;gate_exp=0.916', 'spcop:gate=10;gate_c0=1.290e-06;gate_exp=0.916',
             'rw:1.6', 'rw:1.2', 'rw:2.0', 'rw:3.0', 'spcop:fw=0.03', 'spcop:fw=0.1', 'spcop:ks=12', 'spcop:ks=1', 'spcop:passes=3', 'spcop:rel_skip=0']
    for chi in chis:
        b, brec = M[('svd', chi)]
        for a in order:
            if (a, chi) not in M or a == 'svdt':
                continue
            m, r = M[(a, chi)]
            f, bf = m['final'], b['final']
            cc = f"{r['cut_cpu'] / brec['cut_cpu']:.1f}" if 'cut_cpu' in r and 'cut_cpu' in brec else '-'
            nm = a.replace(';gate_c0=1.290e-06;gate_exp=0.916', ' (c0,exp refit)')
            if a == 'svd':
                print(f"| {chi} | svd (abs rms) | {e(b['dC_Z']['front']['rms'])} | {e(b['dC_Z']['behind']['rms'])} | {e(b['dw']['front']['rms'])} | {e(b['dw']['behind']['rms'])} | {b['lag_Z']['0.5']['max']:.3f} | {bf['infid']:.3f} | {e(bf['far_ge3_rms'])} | {e(b['cz0_rms'])} | {e(bf['win1_rms'])} | {e(bf['win2_rms'])} | {e(bf['win3_rms'])} | 1.0 | - |")
                continue
            fr = f"{r.get('fired', '-')}/{r.get('trunc', '-')}"
            print(f"| {chi} | {nm} | {m['dC_Z']['front']['rms'] / b['dC_Z']['front']['rms']:.2f} | {m['dC_Z']['behind']['rms'] / b['dC_Z']['behind']['rms']:.2f} | {m['dw']['front']['rms'] / b['dw']['front']['rms']:.2f} | {m['dw']['behind']['rms'] / b['dw']['behind']['rms']:.2f} | {m['lag_Z']['0.5']['max']:.3f} | {f['infid'] / bf['infid']:.2f} | {f['far_ge3_rms'] / bf['far_ge3_rms']:.2f} | {m['cz0_rms'] / b['cz0_rms']:.2f} | {f['win1_rms'] / bf['win1_rms']:.2f} | {f['win2_rms'] / bf['win2_rms']:.2f} | {f['win3_rms'] / bf['win3_rms']:.2f} | {cc} | {fr} |")
    print()


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3']
    if '1' in which:
        test1()
    if '2' in which:
        test2()
    if '3' in which:
        test3('ising')
    if 'h' in which:
        test3('heis')
    if 't' in which:
        test3('tfim')
