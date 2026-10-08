"""Markdown tables for the results report from results/a1_*.json (+ supp_*.json).   python rank5/make_tables.py > results/tables.md"""
import sys
import json
import glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_a1 as A  # noqa: E402

RES = HERE / 'results'
recs = A.load(str(RES))
table, summary = A.a1_verdict(recs)
by = {(r['model'], r['b']): r for r in recs}


def cellof(model, b, setname, chi):
    for c in by[(model, b)]['cells']:
        if c['set'] == setname and c['chi'] == chi:
            return c


def e(x):
    return f'{x:.2e}'


print('## A0 gate pre-check (arm (i) cut; ratio = Bres / (c0 (tail/1e-4)^0.65), gate closes if < 1)\n')
print('| model | cut b | set | chi | tail (mean eps_f) | Bres (rms over targets) | cost | ratio | ratio min over targets | ratio max over targets |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in recs:
    for c in r['cells']:
        g = c['gate']
        print(f"| {r['model']} | {r['b']} | {c['set']} | {c['chi']} | {e(g['eq']['tail'])} | {e(g['eq']['bres'])} | {e(g['eq']['cost'])} | {g['eq']['ratio']:.2f} | {g['ratio_min']:.2f} | {g['ratio_max']:.2f} |")

print('\n## A1 E_near by arm (max over targets of the max 1-3-site trace distance inside the fit region)\n')
print('| model | b | set | chi | (i) stack lit | (ii) oracle grid | (ii+) dense oracle | (iii) dressed (prereg a) | (iv) spcf-multi k=0.1 | (v) separate SVD | iv/i | iv/ii | iv/iii | E_out iv/i | max rel fid loss | pass |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k in sorted(table, key=lambda k: (k[0], -k[1], k[2], k[3])):
    m, b, s, chi = k
    c = cellof(m, b, s, chi)
    v = table[k]
    print(f"| {m} | {b} | {s} | {chi} | {e(c['i']['m']['E_near'])} | {e(c['ii']['m']['E_near'])} | {e(c['ii_plus']['m']['E_near'])} | {e(c['iii']['m']['E_near'])} | "
          f"{e(c['iv']['m']['E_near'])} | {e(c['v']['m']['E_near'])} | {v['r_i']:.2f} | {v['r_ii']:.2f} | {v['r_iii']:.2f} | {v['drift']:.2f} | {v['fid']:+.3f} | {'PASS' if v['pass'] else 'no'} |")
