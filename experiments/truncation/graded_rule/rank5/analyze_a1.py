"""A0 and A1 verdicts from results/a1_*.json against the pre-registered criteria of reports/first_moves/rank5_dmrg.md.

    python rank5/analyze_a1.py [results_dir] [--tag _suffix]

Interpretations fixed BEFORE the grid was run (also stated in the results report):
  * cell = (model, cut b, set variant, chi).  Set groups: S2; S3 = {S3p (omega at the peak), S3h (half maximum, high side)};
    S4 = {t=1,2,3}.  A group passes at a (model, cut) if, for a strict majority of its variants, the variant passes in >= 2 of 3 chi.
  * A1 passes at a (model, cut) if >= 2 of the 3 groups pass.  A1 passes overall if it passes at >= 3 of the 4 primary
    configurations (models ising and tfim_crit, cuts b=5 and b=3); ising2 and heis are reported but are not primary.
  * variant/chi cell passes if ALL of: E_near(iv) <= 0.5 E_near(i), <= 0.67 E_near(ii), <= 0.67 E_near(iii);
    per-target fidelity loss: max_f (eps_f(iv) - eps_f(i)) / eps_f(i) <= 0.10 (+1e-3 numerical slack);
    far drift: E_out(iv) <= 1.5 E_out(i), where E_out is the max trace distance over all 1-3-site windows not contained in the fit region.
  * A0 set-level ratio = Bres/cost with equal-weight aggregation over targets (tail = mean eps_f, Bres = rms over targets of the per-target Bres);
    alternatives (literature weights, min/max over targets) are reported.
"""
import sys
import json
import glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY_MODELS = ('ising', 'tfim_crit')
GROUPS = {'S2': ['S2'], 'S3': ['S3p', 'S3h'], 'S4': ['S4t1', 'S4t2', 'S4t3']}


def load(res_dir, tag=''):
    recs = []
    for fn in sorted(glob.glob(str(Path(res_dir) / f'a1_*_N*_b*{tag}.json'))):
        if tag == '' and any(x in fn for x in ('_smoke', '_kap')):
            continue
        r = json.load(open(fn))
        recs.append(r)
    return recs


def cell_flags(c):
    i, ii, iii, iv = (c[k]['m'] for k in ('i', 'ii', 'iii', 'iv'))
    e = lambda m: m['E_near']
    eps_i, eps_iv = np.array(i['eps']), np.array(iv['eps'])
    fl = dict(
        r_i=e(iv) / e(i), r_ii=e(iv) / e(ii), r_iii=e(iv) / e(iii), r_iip=e(iv) / e(c['ii_plus']['m']),
        c1=e(iv) <= 0.5 * e(i), c2=e(iv) <= 0.67 * e(ii), c3=e(iv) <= 0.67 * e(iii),
        fid=float(np.max((eps_iv - eps_i) / np.maximum(eps_i, 1e-14))),
        drift=iv['E_out'] / i['E_out'])
    fl['c4'] = fl['fid'] <= 0.10 + 1e-3
    fl['c5'] = fl['drift'] <= 1.5
    fl['pass'] = bool(fl['c1'] and fl['c2'] and fl['c3'] and fl['c4'] and fl['c5'])
    fl['gain_ii'] = e(ii) / e(iv)          # >1: spcf better than the oracle weight
    fl['gain_iii'] = e(iii) / e(iv)
    fl['gain_i'] = e(i) / e(iv)
    return fl


def a1_verdict(recs, show=True):
    table = {}
    for r in recs:
        for c in r['cells']:
            table[(r['model'], r['b'], c['set'], c['chi'])] = cell_flags(c)
    models = sorted({k[0] for k in table})
    cuts = sorted({k[1] for k in table}, reverse=True)
    chis = sorted({k[3] for k in table})
    summary = {}
    for m in models:
        for b in cuts:
            gp = {}
            for g, vs in GROUPS.items():
                vpass = []
                for v in vs:
                    n = sum(table.get((m, b, v, chi), {}).get('pass', False) for chi in chis)
                    vpass.append(n >= 2)
                gp[g] = sum(vpass) > len(vs) / 2
            summary[(m, b)] = dict(groups=gp, a1=sum(gp.values()) >= 2)
    return table, summary


def kill_checks(table):
    cells = [(k, v) for k, v in table.items() if k[2] in sum((GROUPS[g] for g in GROUPS), []) and k[0] != 'heis']
    gi = np.array([v['gain_ii'] for _, v in cells])
    g3 = np.array([v['gain_iii'] for _, v in cells])
    return dict(n=len(cells), frac_gain_ii_lt13=float(np.mean(gi < 1.3)), max_gain_ii=float(gi.max()),
                frac_gain_iii_lt13=float(np.mean(g3 < 1.3)), max_gain_iii=float(g3.max()),
                kill_oracle_everywhere=bool(np.all(gi < 1.3)), kill_dressed_everywhere=bool(np.all(g3 < 1.3)),
                frac_ii_beats_iv=float(np.mean(gi > 1.0)), frac_iii_beats_iv=float(np.mean(g3 > 1.0)))


def a0_stats(recs):
    rows = []
    for r in recs:
        for c in r['cells']:
            g = c['gate']
            rows.append(dict(model=r['model'], b=r['b'], set=c['set'], chi=c['chi'], ratio_eq=g['eq']['ratio'], ratio_lw=g['lw']['ratio'],
                             ratio_max=g['ratio_max'], ratio_min=g['ratio_min'], tail_eq=g['eq']['tail'], bres_eq=g['eq']['bres'],
                             cost_eq=g['eq']['cost']))
    return rows


def fmt(x, p=2):
    return f'{x:.{p}e}'


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else str(HERE / 'results')
    tag = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
    recs = load(res_dir, tag)
    print(f'{len(recs)} result files: ' + ', '.join(f"{r['model']}/b{r['b']}" for r in recs))
    # ------------------------------------------------------------------ A0
    rows = a0_stats(recs)
    S24 = [r for r in rows if r['set'] != 'S1']
    print('\n== A0: gate pre-check (Bres < cost means the gate would close).  S2-S4 cells, non-heis models vs heis ==')
    for label, sel in (('non-heis', lambda r: r['model'] != 'heis'), ('heis', lambda r: r['model'] == 'heis'), ('all', lambda r: True)):
        rr = [r for r in S24 if sel(r)]
        if not rr:
            continue
        for key in ('ratio_eq', 'ratio_lw', 'ratio_max', 'ratio_min'):
            closed = np.mean([r[key] < 1.0 for r in rr])
            print(f'  {label:9s} {key:9s}: closed in {100 * closed:5.1f}% of {len(rr)} cells; median ratio {np.median([r[key] for r in rr]):.3f}')
    print('\n  closed fraction by chi (ratio_eq), non-heis S2-S4:')
    for chi in sorted({r['chi'] for r in S24}):
        rr = [r for r in S24 if r['model'] != 'heis' and r['chi'] == chi]
        print(f'    chi={chi}: closed {100 * np.mean([r["ratio_eq"] < 1 for r in rr]):5.1f}% of {len(rr)}')
    print('  S1 (negative control) closed fraction (ratio_eq), non-heis: %.1f%%' %
          (100 * np.mean([r['ratio_eq'] < 1 for r in rows if r['set'] == 'S1' and r['model'] != 'heis'])))
    # ------------------------------------------------------------------ A1
    table, summary = a1_verdict(recs)
    print('\n== A1: per-cell E_near ratios (iv/i, iv/ii, iv/iii), fidelity loss, drift ==')
    for k in sorted(table, key=lambda k: (k[0], -k[1], k[2], k[3])):
        v = table[k]
        print(f'  {k[0]:9s} b={k[1]} {k[2]:5s} chi={k[3]}: iv/i {v["r_i"]:.2f} iv/ii {v["r_ii"]:.2f} iv/iii {v["r_iii"]:.2f} iv/ii+ {v["r_iip"]:.2f}'
              f' fid {v["fid"]:+.3f} drift {v["drift"]:.2f}  c1-5 {"".join("Y" if v[c] else "n" for c in ("c1", "c2", "c3", "c4", "c5"))}  {"PASS" if v["pass"] else ""}')
    print('\n== A1 group/config verdicts ==')
    for (m, b), s in sorted(summary.items()):
        print(f'  {m:9s} b={b}: groups {s["groups"]}  -> A1 {"PASS" if s["a1"] else "fail"}')
    prim = [(m, b) for (m, b) in summary if m in PRIMARY_MODELS]
    npass = sum(summary[k]['a1'] for k in prim)
    print(f'  primary configurations passing: {npass} of {len(prim)}  ->  A1 overall: {"PASS" if npass >= 3 else "NOT PASSED"}')
    kc = kill_checks(table)
    print('\n== kill checks (S2-S4 cells, non-heis) ==')
    for k, v in kc.items():
        print(f'  {k}: {v}')
    # ------------------------------------------------------------------ negative control
    print('\n== negative control C: gain E_near(i)/E_near(iv) of S1 vs S2-S4 ==')
    for m in sorted({k[0] for k in table}):
        for b in sorted({k[1] for k in table}, reverse=True):
            for chi in sorted({k[3] for k in table}):
                g1 = table.get((m, b, 'S1', chi), {}).get('gain_i')
                gs = [v['gain_i'] for k, v in table.items() if k[0] == m and k[1] == b and k[3] == chi and k[2] != 'S1']
                if g1 is not None and gs:
                    print(f'  {m:9s} b={b} chi={chi}: S1 gain {g1:.2f}   S2-S4 median {np.median(gs):.2f} max {np.max(gs):.2f}')
    json.dump(dict(a0=rows, a1={f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v for k, v in table.items()},
                   summary={f'{k[0]}|{k[1]}': v for k, v in summary.items()}, kill=kc),
              open(Path(res_dir) / f'a1_analysis{tag}.json', 'w'), default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o), indent=1)


if __name__ == '__main__':
    main()
