"""Test 3 tables and pre-registered verdict from rank1/results/test3_{model}.json.  usage: an_test3.py MODEL [ARM_PREFIX_FOR_VERDICT]"""
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


def fmt(x):
    return f'{x:.2e}'


def run(model, verdict_arms):
    recs = json.load(open(HERE / 'results' / f'test3_{model}.json'))
    ref = B.get_ref(model, N, T, dt)
    M = {}
    for r in recs:
        M[(r['arm'], r['chi'])] = (B.metrics(r, ref), r)
    chis = sorted({r['chi'] for r in recs})
    arms = sorted({r['arm'] for r in recs}, key=lambda a: (a != 'svd', a))
    P(f'=== Test 3: {model}, N={N}, T={T}, dt={dt} ===')
    P(f'exact C_Z max {B.O.C_from_p1(ref["p1"][1:]).max():.3f}')
    for chi in chis:
        P(f'\n-- chi = {chi} (absolute; ratios to svd in brackets) --')
        base, brec = M[('svd', chi)]
        for a in arms:
            m, rec = M[(a, chi)]
            fin = m['final']
            bfin = base['final']
            def rr(x, y):
                return f'{x / y:.2f}' if y > 0 else 'nan'
            P(f"{a:34s} C_Z rms front {fmt(m['dC_Z']['front']['rms'])} [{rr(m['dC_Z']['front']['rms'], base['dC_Z']['front']['rms'])}] "
              f"behind {fmt(m['dC_Z']['behind']['rms'])} [{rr(m['dC_Z']['behind']['rms'], base['dC_Z']['behind']['rms'])}] "
              f"| w front {fmt(m['dw']['front']['rms'])} [{rr(m['dw']['front']['rms'], base['dw']['front']['rms'])}] "
              f"behind {fmt(m['dw']['behind']['rms'])} [{rr(m['dw']['behind']['rms'], base['dw']['behind']['rms'])}] "
              f"| lag0.5 {m['lag_Z']['0.5']['max']:.3f} (svd {base['lag_Z']['0.5']['max']:.3f}) "
              f"| infid {fmt(fin['infid'])} [{rr(fin['infid'], bfin['infid'])}] far>=3 {fmt(fin['far_ge3_rms'])} [{rr(fin['far_ge3_rms'], bfin['far_ge3_rms'])}] "
              f"cz0 {fmt(m['cz0_rms'])} [{rr(m['cz0_rms'], base['cz0_rms'])}] "
              f"| win1 [{rr(fin['win1_rms'], bfin['win1_rms'])}] win2 [{rr(fin['win2_rms'], bfin['win2_rms'])}] win3 [{rr(fin['win3_rms'], bfin['win3_rms'])}]")
    # timing
    P('\n-- cut CPU time (s, process_time inside the cut) and ratio to svd --')
    for chi in chis:
        base = M[('svd', chi)][1].get('cut_cpu')
        for a in arms:
            rec = M[(a, chi)][1]
            if 'cut_cpu' in rec and base:
                P(f"chi={chi:2d} {a:34s} cut cpu {rec['cut_cpu']:.1f}s ratio {rec['cut_cpu'] / base:.1f}  fired {rec.get('fired', '-')}/{rec.get('trunc', '-')} skipped {rec.get('skipped', '-')}")
    # verdict
    for va in verdict_arms:
        P(f'\n=== pre-registered reading for arm {va} on {model} ===')
        okchi = {'front': 0, 'behind': 0}
        details = []
        for chi in chis:
            if (va, chi) not in M:
                continue
            m, rec = M[(va, chi)]
            b, _ = M[('svd', chi)]
            for reg in ('front', 'behind'):
                rc = m['dC_Z'][reg]['rms'] / b['dC_Z'][reg]['rms']
                rw = m['dw'][reg]['rms'] / b['dw'][reg]['rms']
                okchi[reg] += int(rc <= 0.5 and rw <= 0.5)
                details.append((chi, reg, rc, rw))
            lag_b = b['lag_Z']['0.5']['max']
            lag_m = m['lag_Z']['0.5']['max']
            red = (lag_b - lag_m) / lag_b if lag_b > 1e-3 else float('nan')
            fin, bfin = m['final'], b['final']
            P(f"chi={chi:2d}: C_Z ratios front {details[-2][2]:.2f} behind {details[-1][2]:.2f} | w ratios front {details[-2][3]:.2f} behind {details[-1][3]:.2f} | lag0.5 reduction {red:+.0%} (svd {lag_b:.3f}, arm {lag_m:.3f}) | "
              f"infid x{fin['infid'] / bfin['infid']:.2f} far x{fin['far_ge3_rms'] / bfin['far_ge3_rms']:.2f} cz0 x{m['cz0_rms'] / b['cz0_rms']:.2f}")
        P(f"chis with >=2x lower rms in BOTH C_Z and w: front {okchi['front']}/{len(chis)}, behind {okchi['behind']}/{len(chis)}  (success needs >= 2 of 3 in front or behind)")
        worst = max(max(d[2], d[3]) for d in details)
        best = min(min(d[2], d[3]) for d in details if True)
        allgt = all(min(d[2], d[3]) > 0.75 for d in details)
        P(f"kill test (ratio > 0.75 at all chi, both regions, C_Z and w): {all(d[2] > 0.75 and d[3] > 0.75 for d in details)}; best ratio seen {best:.2f}, worst {worst:.2f}")


if __name__ == '__main__':
    model = sys.argv[1]
    run(model, sys.argv[2:])
    open(HERE / 'results' / f'test3_{model}_summary.txt', 'w').write('\n'.join(lines) + '\n')
