"""Ratios to plain SVD at equal chi. python pin_table.py out.json [out2.json ...]"""
import json, sys
rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))
S = {r['chi']: r for r in rows if r['arm'] == 'svd'}
print(f"{'arm':<58}{'chi':>4}{'1site':>7}{'nn':>7}{'nnn':>7}{'rdm2':>7}{'E':>8}{'infid':>7}{'cpu s':>7} {'res':>6} {'cutinf':>6}")
for r in sorted(rows, key=lambda r: (r['chi'], r['arm'] != 'svd', r['arm'])):
    s = S.get(r['chi'])
    if s is None:
        continue
    q = lambda k: r[k] / max(s[k], 1e-300)
    print(f"{r['arm'][:57]:<58}{r['chi']:>4}{q('single_rms'):7.2f}{q('nn_rms'):7.2f}{q('nnn_rms'):7.2f}{q('rdm2'):7.2f}{q('E_abs'):8.2f}{q('infid'):7.2f}"
          f"{r['cpu']:7.0f} {r.get('res_ratio', float('nan')):6.3f} {r.get('cutinf_ratio', float('nan')):6.2f}")
