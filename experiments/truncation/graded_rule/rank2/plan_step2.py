"""Build the job list of Test 1 step 2: in every usable cell (SVD rdm2 in [1e-3, 1e-1] in that gauge) pick up to 4 chi values <= chimax from the window.

    python plan_step2.py scan1.json,scan2.json mus fam T gauges arms out_jobs.json [chimax] [npick]
e.g. python plan_step2.py results/scan_svd_stag.json,results/scan_svd_dw.json 1,0.5,0.25 stag,dw 3,4 plain,back spcf jobs_A.json
"""
import sys, json
import numpy as np
rows = []
for f in sys.argv[1].split(','):
    rows += json.load(open(f))
mus = sys.argv[2].split(',')
fams, Ts, gauges, arms = sys.argv[3].split(','), [float(x) for x in sys.argv[4].split(',')], sys.argv[5].split(','), sys.argv[6].split(',')
out = sys.argv[7]
chimax = int(sys.argv[8]) if len(sys.argv) > 8 else 32
npick = int(sys.argv[9]) if len(sys.argv) > 9 else 4
jobs = []
for fam in fams:
    for mu in mus:
        for T in Ts:
            for g in gauges:
                use = sorted(r['chi'] for r in rows if r['arm'] == 'svd' and r['family'] == fam and str(r['mu']) == str(float(mu) if mu != 'inf' else 'inf')
                             and r['T'] == T and r['gauge'] == g and 1e-3 <= r['rdm2'] <= 1e-1 and r['chi'] <= chimax)
                if len(use) < 2:
                    print('skip (no usable window)', fam, mu, T, g, use)
                    continue
                pick = use if len(use) <= npick else [use[i] for i in sorted({int(round(x)) for x in np.linspace(0, len(use) - 1, npick)})]
                print(fam, mu, T, g, 'window', use, 'pick', pick)
                for chi in sorted(pick, reverse=True):
                    for a in arms:
                        jobs.append([fam, mu, g, T, a, chi])
json.dump(jobs, open(out, 'w'))
print(len(jobs), 'jobs ->', out)
