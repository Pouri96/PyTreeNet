"""Markdown tables for the results report -> results/tables.md"""
import json, glob
import numpy as np
S = json.load(open('results/test1_summary.json'))
out = []
def sub(arm='spcf'):
    return [s for s in S if s['arm'] == arm]
mus = ['0.25', '0.5', '1.0', '2.0', 'inf']
fam_name = {'stag': 'staggered', 'dw': 'domain wall'}
out.append('### T1. Test 1, ungated spcf, median over the SVD-usable chi window (SVD in the same gauge has 1e-3 <= rdm2 <= 1e-1)\n')
out.append('R = SVD in the better of the two gauges / spcf. zz = far ZZ (distance 3, 4) rms spcf / SVD(better gauge). td = physical trace distance spcf / SVD(better gauge). Pass = R(rdm2) >= 2, R(nn) >= 2, zz <= 1.5, td <= 1.10.\n')
out.append('| family | mu | T | spcf gauge | chi run | R rdm2 | R nn | R nnn | zz | td | like-for-like rdm2 | pass |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for fam in ('stag', 'dw'):
    for mu in mus:
        for T in (2.0, 3.0, 4.0):
            for g in ('plain', 'back'):
                s = [x for x in sub() if x['family'] == fam and x['mu'] == mu and x['T'] == T and x['gauge'] == g]
                if not s:
                    continue
                s = s[0]; m = s['med']
                out.append(f"| {fam_name[fam]} | {mu if mu != 'inf' else 'inf'} | {T:g} | {g} | {','.join(map(str, s['chis']))} | {m['R_rdm2']:.2f} | {m['R_nn']:.2f} | {m['R_nnn']:.2f} | {m['zz']:.2f} | {m['td']:.2f} | {m['L_rdm2']:.2f} | {'PASS' if s['success'] else 'no'} |")
out.append('')
# gated
out.append('### T1b. Same for the gated arm (gate 1.0), mu <= 1 and controls, plain gauge only (rows with T=3, 4)\n')
out.append('| family | mu | T | gauge | R rdm2 | R nn | pass |')
out.append('|---|---|---|---|---|---|---|')
for s in sub('spcfg:1.0'):
    if s['gauge'] == 'plain':
        m = s['med']
        out.append(f"| {fam_name[s['family']]} | {s['mu']} | {s['T']:g} | plain | {m['R_rdm2']:.2f} | {m['R_nn']:.2f} | {'PASS' if s['success'] else 'no'} |")
out.append('')
# anchor
A = json.load(open('results/test0b_anchor.json'))
out.append('### T0b. Anchor: mu = inf staggered purification (plain gauge) vs the existing d = 2 code, Ising N = 10, T = 3\n')
out.append('Entries are d4/d2 ratios of the error (1.0000 = identical). max|d2-d4| is the largest absolute difference over the seven metrics.\n')
out.append('| chi | arm | max abs diff | rdm2 | rdm3 | single | nn | nnn | E | infid | fired d2/d4 |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|')
for r in A:
    k = ['rdm2', 'rdm3', 'single_rms', 'nn_rms', 'nnn_rms', 'E_abs', 'infid']
    mx = max(abs(r['d2'][x] - r['d4'][x]) for x in k)
    out.append(f"| {r['chi']} | {r['arm']} | {mx:.1e} | " + ' | '.join(f"{r['d4'][x] / r['d2'][x]:.4f}" for x in k) + f" | {r.get('fired_d2', '')}{'/' if 'fired_d4' in r else ''}{r.get('fired_d4', '')} |")
out.append('')
open('results/tables.md', 'w').write('\n'.join(out))
print('\n'.join(out[:6]))

# condensed table for the report
lines = ['| family | mu | T | plain: R rdm2 / R nn / zz / td | | back: R rdm2 / R nn / zz / td | |', '|---|---|---|---|---|---|---|']
def cell(fam, mu, T, g):
    s = [x for x in sub() if x['family'] == fam and x['mu'] == mu and x['T'] == T and x['gauge'] == g]
    if not s:
        return '-', ''
    m = s[0]['med']
    return f"{m['R_rdm2']:.2f} / {m['R_nn']:.2f} / {m['zz']:.2f} / {m['td']:.2f}", ('PASS' if s[0]['success'] else 'no')
for fam in ('stag', 'dw'):
    for mu in ('0.25', '0.5', '1.0', '2.0', 'inf'):
        for T in (3.0, 4.0):
            p, pv = cell(fam, mu, T, 'plain'); b, bv = cell(fam, mu, T, 'back')
            lines.append(f"| {fam_name[fam]} | {mu} | {T:g} | {p} | {pv} | {b} | {bv} |")
open('results/tables_condensed.md', 'w').write('\n'.join(lines))
