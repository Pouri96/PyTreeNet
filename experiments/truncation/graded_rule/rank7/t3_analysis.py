"""T3 post-processing: spcf (static targets, taus = ()) against SVD on the 1D kicked Ising circuit.

    python rank7/t3_analysis.py GATE_TAG LABEL [LABEL ...]       # files rank7/results/t3_<L>_svd_<chi>.npz, t3_<L>_spcf_<tag>_<chi>.npz

Per setting:
  * errors against the dense reference at every Floquet depth: |dZ_{N/2}|, |d(mean Z)|, 1-site rms (all 3N Paulis), nn rms, infidelity,
    eps_c (all-pairs <Z_i Z_j>, plain and connected);
  * SVD horizon d*(chi, 1e-2): first depth with error > 1e-2 for <Z_{N/2}> (the plan's metric) and for the 1-site rms;
  * r at that depth: chi_SVD,eq(spcf error)/chi, from the SVD ladder, log-log, for the 1-site rms and nn rms (the single-observable error is
    not monotone in chi and cannot be inverted), plus the plain error ratio spcf/SVD of <Z_{N/2}> at equal chi.

Pre-registered reading of the T3 success criterion (before the data were looked at): a setting passes if, over the chi_ref in {8, 12, 16, 24} whose
SVD horizon d*(chi_ref, 1e-2; Z_mid) lies inside depth <= 20 and is at least 2, the MEDIAN r of the 1-site rms at d* is >= 1.5 and the MEDIAN
eps_c ratio (spcf/SVD, plain) over depths d >= 2 with SVD 1-site error >= 1e-4 is <= 1.2.  T3 succeeds if two of the three ergodic settings
(A, B, C) pass.  T3 is killed if the median r < 1.3 in all three.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rutil  # noqa: E402

R = HERE / 'results'
N = 20
DEPTH = 20
SVD_CHIS = [6, 8, 12, 16, 24, 32, 48, 64]
SPCF_CHIS = [8, 12, 16, 24]
iu = np.triu_indices(N, 1)


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != 'meta'}
    d['meta'] = json.loads(str(z['meta']))
    d['F'] = np.exp(d['logF'])
    d['infid'] = 1 - np.abs(d['ov']) ** 2
    return d


def metrics(d, ref):
    o = {}
    o['eZ'] = np.abs(d['z'][:, N // 2] - ref['z'][:, N // 2])
    o['eM'] = np.abs(d['z'].mean(axis=1) - ref['z'].mean(axis=1))
    o['e1'] = np.sqrt(np.mean((d['single'] - ref['single']) ** 2, axis=1))
    o['enn'] = np.sqrt(np.mean((d['nn'] - ref['nn']) ** 2, axis=1))
    c, ct = d['zz'][:, iu[0], iu[1]], ref['zz'][:, iu[0], iu[1]]
    o['epsc'] = np.sqrt(np.sum((c - ct) ** 2, axis=1) / np.sum(ct ** 2, axis=1))
    cc = c - d['z'][:, iu[0]] * d['z'][:, iu[1]]
    cct = ct - ref['z'][:, iu[0]] * ref['z'][:, iu[1]]
    o['epsc_conn'] = np.sqrt(np.sum((cc - cct) ** 2, axis=1) / np.sum(cct ** 2, axis=1))
    o['infid'] = d['infid']
    o['F'] = d['F']
    o['fired'] = d['fired']
    o['calls'] = d['calls']
    return o


def first_above(e, tol):
    idx = np.where(e > tol)[0]
    return int(idx[0] + 1) if len(idx) else None          # depth (1-based)


def analyse(tag, label):
    ref = dict(np.load(R / f't3_ref_obs_{label}_N{N}_d{DEPTH}.npz'))
    svd = {c: metrics(load(R / f't3_{label}_svd_{c}.npz'), ref) for c in SVD_CHIS if (R / f't3_{label}_svd_{c}.npz').exists()}
    spcf = {c: metrics(load(R / f't3_{label}_spcf_{tag}_{c}.npz'), ref) for c in SPCF_CHIS if (R / f't3_{label}_spcf_{tag}_{c}.npz').exists()}
    sc = sorted(svd)
    print(f'\n### setting {label}')
    for key, name in (('eZ', '|dZ_mid|'), ('e1', '1-site rms'), ('epsc', 'eps_c (plain ZZ)'), ('infid', 'infidelity')):
        print(f'\n{name} vs depth')
        print('| arm | ' + ' | '.join(str(d) for d in range(2, DEPTH + 1, 2)) + ' |')
        print('|---|' + '---|' * len(range(2, DEPTH + 1, 2)))
        for c in sc:
            print(f'| svd{c} | ' + ' | '.join(f'{svd[c][key][d - 1]:.1e}' for d in range(2, DEPTH + 1, 2)) + ' |')
        for c in sorted(spcf):
            print(f'| spcf{c} | ' + ' | '.join(f'{spcf[c][key][d - 1]:.1e}' for d in range(2, DEPTH + 1, 2)) + ' |')
    res = dict(label=label, rows=[])
    print('\nhorizons and r at the SVD horizon (chi_ref = chi of both arms)')
    print('| chi_ref | d*(Z_mid,1e-2) SVD | d*(Z_mid) spcf | d*(1site,1e-2) SVD | d*(1site) spcf | r(1site) at d*Z | r(nn) at d*Z | r(1site) at d*(1site) | dZ ratio spcf/SVD at d*Z | eps_c ratio at d*Z | median eps_c ratio | fired frac |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for c in sorted(spcf):
        if c not in svd:
            continue
        dz = first_above(svd[c]['eZ'], 1e-2)
        d1 = first_above(svd[c]['e1'], 1e-2)
        sz = first_above(spcf[c]['eZ'], 1e-2)
        s1 = first_above(spcf[c]['e1'], 1e-2)

        def req(key, d):
            if d is None:
                return None
            ladder = [svd[x][key][d - 1] for x in sc]
            e = spcf[c][key][d - 1]
            ce, fl = rutil.chi_eq(sc, ladder, e)
            return ce / c, fl
        rz1, rznn = req('e1', dz), req('enn', dz)
        r11 = req('e1', d1)
        gp = np.array([svd[c]['e1'][i] >= 1e-4 and i >= 1 for i in range(DEPTH)])
        ratio_eps = spcf[c]['epsc'] / svd[c]['epsc']
        med_eps = float(np.median(ratio_eps[gp])) if gp.any() else float('nan')
        fr = spcf[c]['fired'][-1] / spcf[c]['calls'][-1]

        def fmt(x):
            return '-' if x is None else f'{x[0]:.2f}{"*" if x[1] else ""}'
        dzr = '-' if dz is None else f"{spcf[c]['eZ'][dz - 1] / svd[c]['eZ'][dz - 1]:.2f}"
        er = '-' if dz is None else f'{ratio_eps[dz - 1]:.2f}'
        print(f'| {c} | {dz or ">20"} | {sz or ">20"} | {d1 or ">20"} | {s1 or ">20"} | {fmt(rz1)} | {fmt(rznn)} | {fmt(r11)} | {dzr} | {er} | {med_eps:.2f} | {fr:.3f} |')
        res['rows'].append(dict(chi=c, dz=dz, d1=d1, sz=sz, s1=s1, r1_at_dz=None if rz1 is None else rz1[0], r_nn_at_dz=None if rznn is None else rznn[0],
                                r1_at_d1=None if r11 is None else r11[0], med_eps=med_eps, fired=fr))
    print('\nr(depth): chi_SVD,eq/chi on the 1-site rms (-: both exact to 1e-9; * beyond ladder)')
    print('| chi | ' + ' | '.join(str(d) for d in range(2, DEPTH + 1, 2)) + ' |')
    print('|---|' + '---|' * len(range(2, DEPTH + 1, 2)))
    for c in sorted(spcf):
        row = []
        for d in range(2, DEPTH + 1, 2):
            e = spcf[c]['e1'][d - 1]
            if e < 1e-9:
                row.append('-')
                continue
            ce, fl = rutil.chi_eq(sc, [svd[x]['e1'][d - 1] for x in sc], e)
            row.append(f'{ce / c:.2f}{"*" if fl else ""}')
        print(f'| {c} | ' + ' | '.join(row) + ' |')
    return res


if __name__ == '__main__':
    tag = sys.argv[1]
    allres = [analyse(tag, lab) for lab in sys.argv[2:]]
    json.dump(allres, open(R / f't3_summary_{tag}.json', 'w'), indent=1, default=float)
