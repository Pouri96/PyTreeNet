"""T2 post-processing: spcf against SVD plus free post-processing on the verification horizon.

    python rank7/t2_analysis.py GATE_TAG [GATE_TAG2 ...]     # tags of the spcf runs, files rank7/results/t2_spcf_<tag>_<chi>.npz

Arms (all from the same TEBD runs; nothing is rerun):
  svd            plain SVD at chi
  svdF           SVD x F_MPS             (Mandra):  O <- exp(logF) * O_MPS, F_MPS = product of the per-cut truncation fidelities
  svdFtrue       SVD x |<ref|psi>|^2     (ORACLE, not available in practice: the best any fidelity rescaling with gamma = 1 could do)
  svdlogF[chi]   linear extrapolation of O against log F_MPS to log F = 0 from the (up to) three largest SVD chi <= chi    (Anand)
  svdlogF_top3   the same with the three largest chi of the whole ladder (48, 64, 96)                                       (Anand, literal)
  svdinv[chi]    linear extrapolation of O against 1/chi to 1/chi = 0 from the (up to) three largest SVD chi <= chi        (Tindall)
  svdinv_top3    the same with the three largest chi of the whole ladder
  spcf, spcfF    spcf at chi, and spcf x F_MPS (F_MPS measured on the spcf cuts)
"extrapolated at chi" uses three SVD runs, so it is generous to the post-processing: it costs up to 3 SVD runs of size chi.

Pre-registered definitions (written before the data were looked at, see rank7_benchmarking_results.md):
  t*(arm, tol, metric)  first sample time at which the rms error of `metric` (1-site or nn) exceeds tol; inf if it never does.
  growth phase (chi)    samples t <= t_peak with SVD(chi) nn-rms error >= 1e-4, t_peak the sample of the maximum of the SVD(chi) nn-rms error.
  win fraction          share of growth-phase samples on which err(spcf, chi) < min over {svd, svdF, svdlogF[chi], svdinv[chi]} at the same chi.
  eps_c                 sqrt( sum_{i<j} (c_ij - c~_ij)^2 / sum_{i<j} c~_ij^2 ), c_ij = <Z_i Z_j>  (plain), and the same for the
                        connected c_ij = <Z_i Z_j> - <Z_i><Z_j>.
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
REF = dict(np.load(R / 't2_ref_obs_ising_N20_T8.0_dt0.1_e5.npz'))
TS = REF['t']
NS = len(TS)
SVD_CHIS = [16, 24, 32, 36, 48, 64, 96]
SPCF_CHIS = [16, 24, 32]


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != 'meta'}
    d['meta'] = json.loads(str(z['meta']))
    d['obs'] = np.concatenate([d['single'], d['nn'], d['nnn']], axis=1)
    d['F'] = np.exp(d['logF'])
    d['infid'] = 1 - np.abs(d['ov']) ** 2
    return d


ROBS = np.concatenate([REF['single'], REF['nn'], REF['nnn']], axis=1)
S1, S2 = slice(0, 3 * N), slice(3 * N, 3 * N + 9 * (N - 1))


def errs(obs):
    """rms error of the 1-site and nn Pauli expectation values against the reference, per sample."""
    return dict(e1=np.sqrt(np.mean((obs[:, S1] - ROBS[:, S1]) ** 2, axis=1)), enn=np.sqrt(np.mean((obs[:, S2] - ROBS[:, S2]) ** 2, axis=1)),
                ennn=np.sqrt(np.mean((obs[:, 3 * N + 9 * (N - 1):] - ROBS[:, 3 * N + 9 * (N - 1):]) ** 2, axis=1)))


iu = np.triu_indices(N, 1)


def eps_c(d, connected=False):
    c = d['zz'][:, iu[0], iu[1]]
    ct = REF['zz'][:, iu[0], iu[1]]
    if connected:
        c = c - d['z'][:, iu[0]] * d['z'][:, iu[1]]
        ct = ct - REF['z'][:, iu[0]] * REF['z'][:, iu[1]]
    return np.sqrt(np.sum((c - ct) ** 2, axis=1) / np.sum(ct ** 2, axis=1))


def extrap(runs, xkey):
    """Linear extrapolation of every observable to x = 0, x = log F (per sample) or 1/chi, from `runs` (list of dicts)."""
    if len(runs) == 1:
        return runs[0]['obs']
    if xkey == 'logF':
        x = np.stack([r['logF'] for r in runs])               # (m, ns)
    else:
        x = np.stack([np.full(NS, 1.0 / r['meta']['chi']) for r in runs])
    y = np.stack([r['obs'] for r in runs])                       # (m, ns, nobs)
    xm = x.mean(axis=0)
    dx = x - xm
    var = (dx ** 2).sum(axis=0)
    cov = (dx[:, :, None] * (y - y.mean(axis=0))).sum(axis=0)
    safe = var > 1e-20
    slope = np.where(safe[:, None], cov / np.where(safe, var, 1.0)[:, None], 0.0)
    icpt = y.mean(axis=0) - slope * xm[:, None]
    # guard: if the runs have indistinguishable F (all F ~ 1, early times) the fit is meaningless -> use the largest-chi run
    big = max(range(len(runs)), key=lambda i: runs[i]['meta']['chi'])
    return np.where(safe[:, None], icpt, runs[big]['obs'])


def tstar(e, tol):
    idx = np.where(e > tol)[0]
    return float(TS[idx[0]]) if len(idx) else np.inf


def tstar_interp(e, tol):
    """first crossing, linearly interpolated in log(error) between the two bracketing samples (t = 0 has error 0 -> use the first sample)."""
    idx = np.where(e > tol)[0]
    if not len(idx):
        return np.inf
    j = idx[0]
    if j == 0:
        return float(TS[0])
    a, b = np.log(max(e[j - 1], 1e-300)), np.log(e[j])
    return float(TS[j - 1] + (TS[j] - TS[j - 1]) * (np.log(tol) - a) / (b - a))


def main(tags):
    svd = {c: load(R / f't2_svd_{c}.npz') for c in SVD_CHIS if (R / f't2_svd_{c}.npz').exists()}
    spcf = {t: {c: load(R / f't2_spcf_{t}_{c}.npz') for c in SPCF_CHIS if (R / f't2_spcf_{t}_{c}.npz').exists()} for t in tags}
    chis = sorted(svd)
    out = dict(ts=TS.tolist())

    # ------------------------------------------------------------------ error curves of all arms
    arms = {}                                  # name -> (obs, run)
    for c in chis:
        d = svd[c]
        arms[f'svd{c}'] = d['obs']
        arms[f'svdF{c}'] = d['F'][:, None] * d['obs']
        arms[f'svdFtrue{c}'] = (np.abs(d['ov']) ** 2)[:, None] * d['obs']
        trip = [svd[x] for x in chis if x <= c][-3:]
        arms[f'svdlogF{c}'] = extrap(trip, 'logF')
        arms[f'svdinv{c}'] = extrap(trip, 'inv')
    top3 = [svd[x] for x in chis][-3:]
    arms['svdlogF_top3'] = extrap(top3, 'logF')
    arms['svdinv_top3'] = extrap(top3, 'inv')
    for t, dd in spcf.items():
        for c, d in dd.items():
            arms[f'spcf_{t}_{c}'] = d['obs']
            arms[f'spcfF_{t}_{c}'] = d['F'][:, None] * d['obs']
    E = {k: errs(v) for k, v in arms.items()}
    EC = {}
    for c in chis:
        EC[f'svd{c}'] = (eps_c(svd[c]), eps_c(svd[c], True))
    for t, dd in spcf.items():
        for c, d in dd.items():
            EC[f'spcf_{t}_{c}'] = (eps_c(d), eps_c(d, True))
    INF = {f'svd{c}': svd[c]['infid'] for c in chis}
    FM = {f'svd{c}': svd[c]['F'] for c in chis}
    for t, dd in spcf.items():
        for c, d in dd.items():
            INF[f'spcf_{t}_{c}'] = d['infid']
            FM[f'spcf_{t}_{c}'] = d['F']

    # ------------------------------------------------------------------ report
    def row(vals, fmt='{:.1e}'):
        return ' | '.join(fmt.format(v) for v in vals)

    print('\n## sample times:', TS.tolist())
    for key, name in (('enn', 'nn rms'), ('e1', '1-site rms')):
        print(f'\n### {name} error vs t (rows: arm)')
        print('| arm | ' + ' | '.join(f'{t:g}' for t in TS) + ' |')
        print('|---|' + '---|' * NS)
        names = [f'svd{c}' for c in chis] + ['svdlogF_top3', 'svdinv_top3'] + [k for k in E if k.startswith('spcf')]
        for k in names:
            print(f'| {k} | ' + row(E[k][key]) + ' |')
    print('\n### infidelity and F_MPS (final sample t=8.0)')
    print('| arm | infid | 1-F_MPS | eps_c (plain) | eps_c (conn) |')
    print('|---|---|---|---|---|')
    for k in [f'svd{c}' for c in chis] + [k for k in INF if k.startswith('spcf')]:
        print(f'| {k} | {INF[k][-1]:.3e} | {1 - FM[k][-1]:.3e} | {EC[k][0][-1]:.2e} | {EC[k][1][-1]:.2e} |')

    # ------------------------------------------------------------------ horizons
    res = {}
    print('\n### verification horizon t*(tol): first sample time with error > tol (inf = never within T = 8); [interp] in brackets')
    print('| arm | metric | tol 1e-2 | tol 1e-3 |')
    print('|---|---|---|---|')
    for k in [f'svd{c}' for c in chis] + ['svdlogF_top3', 'svdinv_top3'] + [k for k in E if k.startswith('spcf')] + [f'svdF{c}' for c in chis] + [f'svdFtrue{c}' for c in chis] + [f'svdlogF{c}' for c in chis] + [f'svdinv{c}' for c in chis]:
        for key, name in (('enn', 'nn'), ('e1', '1site')):
            res[(k, name)] = {tol: (tstar(E[k][key], tol), tstar_interp(E[k][key], tol)) for tol in (1e-2, 1e-3)}
            print(f"| {k} | {name} | {res[(k, name)][1e-2][0]:g} [{res[(k, name)][1e-2][1]:.2f}] | {res[(k, name)][1e-3][0]:g} [{res[(k, name)][1e-3][1]:.2f}] |")

    # ------------------------------------------------------------------ T2 criteria
    summary = {}
    for t, dd in spcf.items():
        print(f'\n## T2 criteria for gate {t}')
        crit_a, crit_b, crit_c = {}, {}, {}
        print('\n(a) t*_spcf(chi) >= t*_SVD(1.5 chi), per metric and tolerance (first-sample t*; the interpolated t* in brackets)')
        print('| chi | 1.5chi | metric | tol | t*_spcf | t*_SVD(1.5chi) | holds |')
        print('|---|---|---|---|---|---|---|')
        ok_all = True
        for c in SPCF_CHIS:
            if c not in dd or int(round(1.5 * c)) not in svd:
                continue
            c15 = int(round(1.5 * c))
            for name in ('nn', '1site'):
                for tol in (1e-2, 1e-3):
                    a, ai = res[(f'spcf_{t}_{c}', name)][tol]
                    b, bi = res[(f'svd{c15}', name)][tol]
                    hold = a >= b
                    ok_all &= hold
                    crit_a[(c, name, tol)] = hold
                    print(f'| {c} | {c15} | {name} | {tol:g} | {a:g} [{ai:.2f}] | {b:g} [{bi:.2f}] | {"yes" if hold else "NO"} |')
        print(f'(a) all cases hold: {ok_all}   ({sum(crit_a.values())}/{len(crit_a)})')

        print('\n(b) win fraction over growth-phase samples vs the best post-processed SVD at equal chi (and vs plain SVD)')
        print('| chi | metric | growth samples | t_peak | win vs best-pp | win vs svd | win(spcf or spcfF) vs best-pp | "within 20%" fraction (spcf > 0.8 best-pp) | win vs best-pp, whole window with SVD err>=1e-4 | win vs best-pp incl. ORACLE svdFtrue |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for c in SPCF_CHIS:
            if c not in dd:
                continue
            tp = int(np.argmax(E[f'svd{c}']['enn']))
            gp = np.array([i <= tp and E[f'svd{c}']['enn'][i] >= 1e-4 for i in range(NS)])
            allw = E[f'svd{c}']['enn'] >= 1e-4
            for key, name in (('enn', 'nn'), ('e1', '1site')):
                pp = np.minimum.reduce([E[f'svd{c}'][key], E[f'svdF{c}'][key]] + ([E[f'svdlogF{c}'][key], E[f'svdinv{c}'][key]] if c in (24, 32, 36, 48, 64, 96) else []))
                sp = E[f'spcf_{t}_{c}'][key]
                spF = np.minimum(sp, E[f'spcfF_{t}_{c}'][key])
                w = (sp < pp)[gp].mean()
                w_svd = (sp < E[f'svd{c}'][key])[gp].mean()
                w_F = (spF < pp)[gp].mean()
                within = (sp > 0.8 * pp)[gp].mean()
                w_all = (sp < pp)[allw].mean()
                crit_b[(c, name)] = w
                w_or = (sp < np.minimum(pp, E[f'svdFtrue{c}'][key]))[gp].mean()
                print(f'| {c} | {name} | {int(gp.sum())} | {TS[tp]:g} | {w:.2f} | {w_svd:.2f} | {w_F:.2f} | {within:.2f} | {w_all:.2f} | {w_or:.2f} |')
        print('(b) >= 0.70 at every chi and metric:', all(v >= 0.7 for v in crit_b.values()))

        print('\n(c) eps_c ratio spcf/SVD at equal chi (median / max over growth-phase samples; final sample); plain and connected')
        print('| chi | median plain | max plain | final plain | median conn | max conn | final conn |')
        print('|---|---|---|---|---|---|---|')
        for c in SPCF_CHIS:
            if c not in dd:
                continue
            tp = int(np.argmax(E[f'svd{c}']['enn']))
            gp = np.array([i <= tp and E[f'svd{c}']['enn'][i] >= 1e-4 for i in range(NS)])
            rp = EC[f'spcf_{t}_{c}'][0] / EC[f'svd{c}'][0]
            rc = EC[f'spcf_{t}_{c}'][1] / EC[f'svd{c}'][1]
            crit_c[c] = (float(np.median(rp[gp])), float(np.median(rc[gp])))
            print(f'| {c} | {np.median(rp[gp]):.2f} | {rp[gp].max():.2f} | {rp[-1]:.2f} | {np.median(rc[gp]):.2f} | {rc[gp].max():.2f} | {rc[-1]:.2f} |')

        print('\nr(t): chi_SVD(t)/chi at equal nn error (ladder svd16..96, log-log; * = beyond the ladder, lower bound)')
        print('| chi | ' + ' | '.join(f'{x:g}' for x in TS) + ' |')
        print('|---|' + '---|' * NS)
        for c in SPCF_CHIS:
            if c not in dd:
                continue
            rr = []
            for i in range(NS):
                ladder_e = [E[f'svd{x}']['enn'][i] for x in chis]
                if E[f'spcf_{t}_{c}']['enn'][i] < 1e-7:            # both arms exact to rounding, r undefined
                    rr.append('-')
                    continue
                ce, fl = rutil.chi_eq(chis, ladder_e, E[f'spcf_{t}_{c}']['enn'][i])
                rr.append(f'{ce / c:.2f}{"*" if fl else ""}')
            print(f'| {c} | ' + ' | '.join(rr) + ' |')
        summary[t] = dict(a_cases=sum(crit_a.values()), a_total=len(crit_a), b={f'{k[0]}_{k[1]}': v for k, v in crit_b.items()},
                          c={str(k): v for k, v in crit_c.items()})
    json.dump(dict(summary=summary), open(R / 't2_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
