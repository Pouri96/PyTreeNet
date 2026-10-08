"""Gain table of the kappa extension: svd vs spcf (a=1) at kappa in {4,8}.  python analyze_kappa.py file.jsonl"""
import _p6  # noqa: F401
import json, sys
R = {}
for l in open(sys.argv[1]):
    r = json.loads(l); R[r['job']] = r
keys = sorted({(r['gamma'], r['kappa'], r['chi']) for r in R.values()})
print('gamma kap chi | nn_svd   nn_spcf  gain | nnn gain | sgl gain | tn ratio | far3 far4 ratio | E ratio(spcf/svd) | fired')
for g, k, c in keys:
    s, p = R.get(f'svd_g{g:g}_chi{c}_k{k}'), R.get(f'spcf_g{g:g}_chi{c}_k{k}')
    if not (s and p) or s['nn_rms'] < 1e-10:
        continue
    print(f"{g:<5g} {k:2d} {c:3d} | {s['nn_rms']:.2e} {p['nn_rms']:.2e} {s['nn_rms']/p['nn_rms']:5.2f} | {s['nnn_rms']/p['nnn_rms']:5.2f} | {s['single_rms']/p['single_rms']:5.2f} | "
          f"{p['trace_norm']/s['trace_norm']:.3f} | {p['far3_rms']/s['far3_rms']:.2f} {p['far4_rms']/s['far4_rms']:.2f} | {p['E_abs']/s['E_abs']:.2f} | {p['fired']}/{p['calls']}")
