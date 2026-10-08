"""Test 2 analysis.   python analyze_test2.py test2.json [spcf_arm=spcf_plain]
Success: spcf-purification error <= DMT error / 1.5 on BOTH rdm2 and nnn_rms at >= 3 of the chi values.  Kill: DMT <= spcf on rdm2 at every chi."""
import sys, json
import numpy as np
rows = json.load(open(sys.argv[1]))
arm = sys.argv[2] if len(sys.argv) > 2 else 'spcf_plain'
chis = sorted({r['chi'] for r in rows})
d = {(r['arm'], r['chi']): r for r in rows}
arms = ['svd_plain', 'svd_back', 'spcf_plain', 'spcf_back', 'spcfg_plain', 'spcfg_back', 'dmt', 'dmt+8', 'frob']
cell = f"{rows[0]['family']} mu={rows[0]['mu']} T={rows[0]['T']:g} N={rows[0]['N']}"
print('cell', cell)
for m, lab in (('rdm2', 'rdm2'), ('nn_rms', 'nn rms'), ('nnn_rms', 'nnn rms'), ('rdm3', 'rdm3'), ('single_rms', 'single rms'), ('zzfar', 'far ZZ (d=3,4) rms'), ('E_abs', '|dE|'), ('tdist', 'trace distance'), ('lam_min', 'lambda_min(rho)'), ('trace', 'tr rho (before normalisation; purification = 1)')):
    print(f'\n{lab}')
    print('  arm'.ljust(14) + ''.join(f'chi={c:<3d}'.rjust(11) for c in chis))
    for a in arms:
        vals = []
        for c in chis:
            r = d.get((a, c))
            vals.append('' if r is None or m not in r else f"{r[m]:.3e}")
        print(f'  {a:12s}' + ''.join(v.rjust(11) for v in vals))
print(f'\nspcf arm judged: {arm}   (params per bulk site: {4} chi^2 for every arm)')
print('  chi | rdm2: dmt/spcf  | nnn: dmt/spcf | nn: dmt/spcf | rdm3 dmt/spcf | zzfar dmt/spcf | |dE| dmt/spcf | pass(>=1.5 on rdm2 and nnn)')
npass, nkill = 0, 0
for c in chis:
    s, m = d[(arm, c)], d[('dmt', c)]
    rr = {k: m[k] / s[k] for k in ('rdm2', 'nnn_rms', 'nn_rms', 'rdm3', 'zzfar', 'E_abs')}
    ok = rr['rdm2'] >= 1.5 and rr['nnn_rms'] >= 1.5
    npass += ok
    nkill += (m['rdm2'] <= s['rdm2'])
    print(f"  {c:3d} |   {rr['rdm2']:7.2f}      |   {rr['nnn_rms']:7.2f}     |  {rr['nn_rms']:7.2f}     |  {rr['rdm3']:7.2f}      |  {rr['zzfar']:7.2f}       |  {rr['E_abs']:7.2f}     | {'yes' if ok else 'no'}")
print(f'  passes at {npass} of {len(chis)} chi values (success needs >= 3 of 4); DMT <= spcf on rdm2 at {nkill} of {len(chis)} chi (kill if all)')
