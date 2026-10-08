"""Test 1 analysis: spcf / gated spcf against SVD on the mu dial, against SVD in the BETTER of the two gauges, pre-registered criteria.

    python analyze_test1.py scan.json[,scan2.json] step2.json[,step2b.json] [out.json]

For every (family, mu, T) and every chi that was run with spcf:
    svd_best(m, chi) = min over gauges of the SVD error m                       (m in rdm2, nn_rms; also zzfar, tdist)
    ratio_g(m, chi)  = svd_best(m, chi) / spcf_g(m, chi)                          g = gauge the spcf arm ran in
    like_g(m, chi)   = svd_g(m, chi)    / spcf_g(m, chi)                          like for like
A (cell, gauge, arm) is judged on the chi values in the SVD usable window (1e-3 <= svd_g rdm2 <= 1e-1, SVD in the same gauge):
    success if median ratio_g(rdm2) >= 2 and median ratio_g(nn) >= 2 and median zzfar_spcf/zzfar_svd_best <= 1.5 and median tdist_spcf/tdist_svd_best <= 1.1.
"""
import sys, json
import numpy as np
from collections import defaultdict

def load(s):
    rows = []
    for f in s.split(','):
        rows += json.load(open(f))
    return rows

scan, step2 = load(sys.argv[1]), load(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
rows = scan + step2
key = lambda r: (r['family'], str(r['mu']), r['T'])
svd = {}
for r in rows:
    if r['arm'] == 'svd':
        svd[(key(r), r['gauge'], r['chi'])] = r
sp = defaultdict(dict)
for r in rows:
    if r['arm'] != 'svd':
        sp[(key(r), r['gauge'], r['arm'])][r['chi']] = r
MET = ('rdm2', 'nn_rms', 'nnn_rms', 'zzfar', 'tdist')
summary = []
def fmt(x): return f'{x:6.2f}'
print(f"{'cell':20s} {'gauge':5s} {'arm':8s} {'chi':>4s} | svd_g rdm2 | svd_best rdm2 (gauge) | spcf rdm2 | R_best(rdm2) R_best(nn) R_best(nnn) | like(rdm2) | zzfar sp/best | tdist sp/best")
for (cell, g, arm), d in sorted(sp.items(), key=lambda kv: (kv[0][0][0], -1 if kv[0][0][1] == 'inf' else -float(kv[0][0][1]), kv[0][0][2], kv[0][1], kv[0][2])):
    res = {m: [] for m in ('R_rdm2', 'R_nn', 'R_nnn', 'L_rdm2', 'L_nn', 'zz', 'td', 'tdL')}
    usable_chis = []
    for chi in sorted(d):
        s_g = svd.get((cell, g, chi))
        s_o = svd.get((cell, 'back' if g == 'plain' else 'plain', chi))
        if s_g is None:
            continue
        best = {m: min(s_g[m], s_o[m]) if s_o else s_g[m] for m in MET}
        bg = 'plain' if (s_o is None or s_g['rdm2'] <= s_o['rdm2']) else ('back' if g == 'plain' else 'plain')
        r = d[chi]
        usable = 1e-3 <= s_g['rdm2'] <= 1e-1
        line = (f"{cell[0]}:{cell[1]}:T{cell[2]:g}".ljust(20) + f" {g:5s} {arm:8s} {chi:4d} | {s_g['rdm2']:9.2e}  | {best['rdm2']:9.2e} ({bg})        | {r['rdm2']:9.2e} | "
                f"{best['rdm2'] / r['rdm2']:6.2f} {best['nn_rms'] / r['nn_rms']:6.2f} {best['nnn_rms'] / r['nnn_rms']:6.2f} | {s_g['rdm2'] / r['rdm2']:6.2f} | "
                f"{r['zzfar'] / best['zzfar']:6.2f} | {r['tdist'] / best['tdist']:6.2f}" + ('' if usable else '   (outside window)'))
        print(line)
        if usable:
            usable_chis.append(chi)
            res['R_rdm2'].append(best['rdm2'] / r['rdm2']); res['R_nn'].append(best['nn_rms'] / r['nn_rms']); res['R_nnn'].append(best['nnn_rms'] / r['nnn_rms'])
            res['L_rdm2'].append(s_g['rdm2'] / r['rdm2']); res['L_nn'].append(s_g['nn_rms'] / r['nn_rms'])
            res['zz'].append(r['zzfar'] / best['zzfar']); res['td'].append(r['tdist'] / best['tdist']); res['tdL'].append(r['tdist'] / s_g['tdist'])
    if usable_chis:
        med = {k: float(np.median(v)) for k, v in res.items()}
        ok = med['R_rdm2'] >= 2 and med['R_nn'] >= 2 and med['zz'] <= 1.5 and med['td'] <= 1.1
        summary.append(dict(family=cell[0], mu=cell[1], T=cell[2], gauge=g, arm=arm, chis=usable_chis, med=med, success=bool(ok)))
print()
print('SUMMARY over the usable window (median over the chi values run inside the window)')
print(f"{'cell':20s} {'gauge':5s} {'arm':8s} {'chis':18s} | R_best rdm2  nn  nnn | like rdm2  nn | zzfar sp/best | tdist sp/best | tdist sp/svd_g | pass")
for s in summary:
    m = s['med']
    print(f"{s['family']}:{s['mu']}:T{s['T']:g}".ljust(20) + f" {s['gauge']:5s} {s['arm']:8s} {str(s['chis']):18s} | {m['R_rdm2']:6.2f} {m['R_nn']:6.2f} {m['R_nnn']:6.2f} | {m['L_rdm2']:6.2f} {m['L_nn']:6.2f} | {m['zz']:6.2f} | {m['td']:6.2f} | {m['tdL']:6.2f} | {'PASS' if s['success'] else 'no'}")
if out:
    json.dump(summary, open(out, 'w'), indent=1)
