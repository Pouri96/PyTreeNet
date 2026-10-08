"""T1 scoring.  Reads the outputs of run_t1.sh (pareto_spcf.py, as the plan specifies) and of run_t1_cold.sh (t1_cold.py, same gate
state for error and timing), converts the errors to r against the chi <= 192 SVD ladder of matched_hp.json, and prints the tables.

    python rank7/t1_score.py            # writes rank7/results/t1_scores.json and prints markdown tables
"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rutil  # noqa: E402

R = HERE / 'results'
CONFIGS = [('1e-2', '1e-7'), ('0.1', '1e-7'), ('0.3', '1e-7'), ('0.6', '1e-7'), ('0.3', '1e-5')]
lad = rutil.svd_ladder()
LC = [r['chi'] for r in lad]


def timing_curve(files):
    """chi -> list of steady cpu of the SVD arm over the given pareto_spcf runs (SVD only)."""
    d = {}
    for f in files:
        if not f.exists():
            continue
        for r in json.load(open(f)):
            if r['arm'] == 'svd':
                d.setdefault(r['chi'], []).append(r['cpu_steady'])
    return d


def interp_time(curve, chi):
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    return float(np.exp(np.interp(np.log(chi), np.log(xs), np.log(ys))))


pre = timing_curve([R / 't1_svdtime_pre.json'])
post = timing_curve([R / 't1_svdtime_post.json'])
both = {c: pre.get(c, []) + post.get(c, []) for c in set(pre) | set(post)}
tcur = {c: min(v) for c, v in both.items()}
tcur_pre = {c: min(v) for c, v in pre.items()}
tcur_post = {c: min(v) for c, v in post.items()}


def score_row(r, svd32_same_file, label):
    out = dict(label=label, chi=r['chi'])
    for m, key in (('nn_rms', 'nn'), ('single_rms', '1site'), ('infid', 'infid')):
        ce, flag = rutil.chi_eq(LC, [x[m] for x in lad], r[m])
        out['r_' + key] = ce / r['chi']
        out['r_' + key + '_flag'] = flag
        out['chi_eq_' + key] = ce
    return out


rows_spec, rows_cold = [], []
for rs, eps in CONFIGS:
    label = f'rs={rs} eps={eps}'
    f = R / f't1_rs{rs}_eps{eps}.json'
    if f.exists():
        d = json.load(open(f))
        svd32 = [x for x in d if x['arm'] == 'svd'][0]['cpu_steady']
        for r in d:
            if r['arm'] != 'spcf':
                continue
            o = score_row(r, svd32, label)
            o.update(f=r['fired'] / r['calls'], cpu_spcf=r['cpu_steady'], cpu_spcf_first=r['cpu'], svd32_in_file=svd32,
                     nn=r['nn_rms'], single=r['single_rms'], infid=r['infid'])
            for key in ('nn', '1site'):
                tsvd = interp_time(tcur, o['chi_eq_' + key])
                o['cpu_svd_eq_' + key] = tsvd
                o['ratio_' + key] = r['cpu_steady'] / tsvd
                o['pred_' + key] = (1 + 15 * o['f']) / o['r_' + key] ** 3
                lam = svd32 / interp_time(tcur, 32)
                o['ratio_loadnorm_' + key] = (r['cpu_steady'] / lam) / tsvd
            rows_spec.append(o)
    fc = R / f't1c_rs{rs}_eps{eps}.json'
    if fc.exists():
        for r in json.load(open(fc)):
            o = score_row(r, None, label)
            o.update(f=r['fired'] / r['calls'], f_reps=[x / r['calls'] for x in r['fired_reps']], cpu_spcf=r['cpu_spcf'], nn=r['nn_rms'],
                     single=r['single_rms'], infid=r['infid'], cpu_svd_same=r['cpu_svd_same'])
            for key, ck in (('nn', 'cpu_svd_eq_nn'), ('1site', 'cpu_svd_eq_1site')):
                o['cpu_svd_eq_' + key] = r[ck]
                o['chi_svd_' + key] = r['chi_svd_' + key]
                o['ratio_' + key] = r['cpu_spcf'] / r[ck]
                o['pred_' + key] = (1 + 15 * o['f']) / o['r_' + key] ** 3
            o['ratio_same_chi'] = r['cpu_spcf'] / r['cpu_svd_same']
            rows_cold.append(o)


def table(rows, cold):
    print('| gate | chi | r nn | r 1-site | r infid | f | nn err | CPU spcf (s) | CPU SVD@eq nn (s) | ratio nn | ratio 1-site | pred (1+15f)/r^3 (nn) | spcf/SVD same chi |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for o in rows:
        same = f"{o['ratio_same_chi']:.1f}" if cold else ''
        print(f"| {o['label']} | {o['chi']} | {o['r_nn']:.2f}{'*' if o['r_nn_flag'] else ''} | {o['r_1site']:.2f}{'*' if o['r_1site_flag'] else ''} | "
              f"{o['r_infid']:.2f} | {o['f']:.3f} | {o['nn']:.2e} | {o['cpu_spcf']:.1f} | {o['cpu_svd_eq_nn']:.1f} | {o['ratio_nn']:.1f} | "
              f"{o['ratio_1site']:.1f} | {o['pred_nn']:.1f} | {same} |")


print('\nAS SPECIFIED (pareto_spcf.py: error from the 1st run, steady time from later runs of the same SPCFast object; f = fired/calls averaged over runs)')
table(rows_spec, False)
print('\nSVD timing ladder (steady cpu, min of 3 reps): pre / post scan')
for c in sorted(tcur):
    print(f'  chi={c}: pre {tcur_pre.get(c, float("nan")):.2f}  post {tcur_post.get(c, float("nan")):.2f}')
print('\nCOLD GATE (t1_cold.py: fresh SPCFast per run; SVD cost measured at the equal-error chi, interleaved)')
table(rows_cold, True)
json.dump(dict(spec=rows_spec, cold=rows_cold, svd_time_pre=tcur_pre, svd_time_post=tcur_post), open(R / 't1_scores.json', 'w'), indent=1, default=float)
