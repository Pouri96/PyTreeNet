"""Ratios (arm / svd at equal chi) of the 2-site, 1-site, nn error and infidelity, and cpu, for every results json given on the command line."""
import json, sys
for f in sys.argv[1:]:
    rows = json.load(open(f))
    sv = {r['chi']: r for r in rows if r['arm'] == 'svd'}
    print('==', f)
    for r in rows:
        if r['arm'] == 'svd':
            print(f"  svd chi={r['chi']}  rdm2={r['rdm2']:.3e} 1site={r['single_rms']:.3e} nn={r['nn_rms']:.3e} infid={r['infid']:.3e} cpu={r['cpu']}")
            continue
        s = sv[r['chi']]
        print(f"  {r['arm'][5:]:42s} rdm2 {r['rdm2'] / s['rdm2']:.2f} 1site {r['single_rms'] / s['single_rms']:.2f} nn {r['nn_rms'] / s['nn_rms']:.2f} "
              f"infid {r['infid'] / s['infid']:.2f} E {r['E_abs'] / s['E_abs']:.2f} cpu x{r['cpu'] / max(s['cpu'], 1e-9):.0f} fired {r.get('fired')}")
