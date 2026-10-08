"""One line per Test 2 cell.   python test2_summary.py results/test2_*.json"""
import sys, json, glob
import numpy as np
print('cell'.ljust(22), 'spcf gauge', ' | DMT/spcf rdm2 at chi=12,16,24,32      | DMT/spcf nnn at chi=12,16,24,32      | pass | DMT/spcf zzfar | DMT/spcf |dE| | SVD_best/spcf rdm2 | DMT lam_min (chi=24)')
for f in sorted(sys.argv[1:]):
    rows = json.load(open(f))
    g = rows[0]['spcf_gauge']
    d = {(r['arm'], r['chi']): r for r in rows}
    chis = sorted({r['chi'] for r in rows})
    arm = f'spcf_{g}'
    rr = {k: [d[('dmt', c)][k] / d[(arm, c)][k] for c in chis] for k in ('rdm2', 'nnn_rms', 'zzfar', 'E_abs')}
    sb = [min(d[('svd_plain', c)]['rdm2'], d[('svd_back', c)]['rdm2']) / d[(arm, c)]['rdm2'] for c in chis]
    npass = sum(a >= 1.5 and b >= 1.5 for a, b in zip(rr['rdm2'], rr['nnn_rms']))
    r0 = rows[0]
    cell = f"{r0['family']} mu={r0['mu']} T={r0['T']:g}"
    print(cell.ljust(22), g.ljust(10), ' | ' + ' '.join(f'{x:6.2f}' for x in rr['rdm2']) + '  | ' + ' '.join(f'{x:6.2f}' for x in rr['nnn_rms']) + '  | ' + f'{npass}/4'
          + ' | ' + ' '.join(f'{x:5.2f}' for x in rr['zzfar']) + ' | ' + ' '.join(f'{x:6.3f}' for x in rr['E_abs']) + ' | ' + ' '.join(f'{x:5.2f}' for x in sb) + ' | ' + f"{d[('dmt', 24)]['lam_min']:.1e}")
