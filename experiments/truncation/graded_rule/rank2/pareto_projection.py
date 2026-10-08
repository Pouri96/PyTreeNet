"""Projection of the N = 10 equal-error comparison to longer chains and bulk bonds (a MODEL, labelled as such in the report; not a measurement).

    python pareto_projection.py   reads results/pareto_points.json (N = 10 runs: error, params, fired fraction) and results/cost_chunk.json (per-cut time and
                                  transient memory at N = 16 bulk bonds), writes results/pareto_projection.json and results/pareto_projection.txt

For every spcf point (variant V, chi_s) of a cell: chi_eq = bond dimension at which the better-gauge SVD reaches the same final rdm2 (log-log
interpolation of the monotone envelope; flagged if outside the ladder).  Then, for a chain of N sites with bulk bonds at chi:
  peak(V, chi_s)   = stored MPS (N sites at the bulk size 4 chi^2 complex numbers, plus the N = 10 edge profile) + 2 theta (two-site tensor before/after the gate)
                     + transient of one cut at bulk bonds (cost_chunk, N = 16, resident-set growth) + constant tables of the cut
  peak(SVD, chi_eq) likewise with the SVD cut.
  time(V, chi_s)   = per cut  t_svd(chi_s) + f * (t_V(chi_s) - t_svd(chi_s)),  f = fired fraction measured at N = 10;   time(SVD, chi_eq) = t_svd(chi_eq)
Ratios < 1 favour spcf.  The per-cut numbers are interpolated log-log in chi (measured at 16, 32, 48, 64, 96, 128).
"""
import os
import json
import numpy as np
import _p2  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MB = 2 ** 20
CELLS = ['stag_mu1', 'stag_mu0.5', 'stag_mu0.25', 'dw_mu1', 'dw_mu0.5', 'dw_mu0.25']
# N = 10 variant key -> per-cut variant in cost_chunk.json
MAP = {'spcf-a1': 'a1', 'spcf-a2': 'a2'}


def cost_interp(rows, variant, what):
    d = {}
    for r in rows:
        if r['variant'] == variant and r['what'] == what:
            d[r['chi']] = r['t_med'] if what == 'time' else r['peak']
    chis = sorted(d)
    if len(chis) < 2:
        return None
    x, y = np.log(chis), np.log([d[c] for c in chis])
    return lambda chi: float(np.exp(np.interp(np.log(chi), x, y) if chis[0] <= chi <= chis[-1] else
                                    np.polyval(np.polyfit(x[-3:], y[-3:], 1), np.log(chi)) if chi > chis[-1] else np.polyval(np.polyfit(x[:3], y[:3], 1), np.log(chi))))


def main():
    pts = json.load(open(os.path.join(RES, 'pareto_points.json')))
    rows = json.load(open(os.path.join(RES, 'cost_chunk.json')))
    tab = {}
    for arm in {k.split('|')[0] for c in pts.values() for k in c}:
        pass
    svd_t = cost_interp(rows, 'svd', 'time')
    svd_m = cost_interp(rows, 'svd', 'rss')
    tables = {}
    out = dict(rows=[])
    lines = []
    Ns = (10, 20, 50, 100)
    for cell in CELLS:
        P = {tuple([k.split('|')[0], int(k.split('|')[1])]): v for k, v in pts[cell].items()}
        # SVD envelope in chi (better gauge)
        chis = sorted({c for (a, c) in P if a == 'svd_plain' and (('svd_back', c) in P)})
        err = np.array([min(P[('svd_plain', c)]['err'], P[('svd_back', c)]['err']) for c in chis])
        env = np.minimum.accumulate(err)
        par = np.array([P[('svd_plain', c)]['params'] if P[('svd_plain', c)]['err'] <= P[('svd_back', c)]['err'] else P[('svd_back', c)]['params'] for c in chis], float)
        for (arm, chi_s), p in sorted(P.items()):
            if arm.startswith('svd') or arm == 'dmt' or 'rss' not in p:
                continue
            e = p['err']
            if env[-1] > e:
                ce, flag = chis[-1], 'above'
            elif env[0] <= e:
                ce, flag = chis[0], 'below'
            else:
                i = int(np.argmax(env <= e))
                f = (np.log(e) - np.log(env[i - 1])) / (np.log(env[i]) - np.log(env[i - 1]))
                ce, flag = float(np.exp(np.log(chis[i - 1]) + f * (np.log(chis[i]) - np.log(chis[i - 1])))), 'ok'
            # per-cut variant for the N = 16 cost tables
            a = arm.split('-')
            var = ('a' + a[1][1:]) if a[0] == 'spcf' else ('a' + a[1][1:] + 'c' + a[2][1:] + ('f' if 'wf' in a else ''))
            tm = cost_interp(rows, var, 'time') or cost_interp(rows, var.rstrip('f'), 'time')
            mm = cost_interp(rows, var, 'rss') or cost_interp(rows, var.rstrip('f'), 'rss')
            if tm is None or mm is None:
                continue
            tab_mb = p.get('tables_MB', 0.0)
            f_fire = (p.get('fired') or 0) / max(p.get('calls') or 1, 1)
            for N in Ns:
                def stored(chi, N=N):
                    # bulk site 4 chi^2 complex128 (16 B); N = 10 edge profile approximated by the measured params at chi (par interpolation) plus (N-10) bulk sites
                    p10 = float(np.exp(np.interp(np.log(chi), np.log(chis), np.log(par)))) if chis[0] <= chi <= chis[-1] else 4.0 * chi * chi * 8
                    return p10 * 16 + (N - 10) * 4 * chi * chi * 16
                th = lambda chi: 16 * chi * chi * 16
                ps = stored(chi_s) + 2 * th(chi_s) + mm(chi_s) + tab_mb * MB
                pv = stored(ce) + 2 * th(ce) + svd_m(ce)
                ts = svd_t(chi_s) + f_fire * (tm(chi_s) - svd_t(chi_s))
                tv = svd_t(ce)
                out['rows'].append(dict(cell=cell, arm=arm, chi=chi_s, N=N, chi_eq=ce, flag=flag, mem_ratio=ps / pv, time_ratio=ts / tv, f_fire=f_fire, peak_spcf_MB=ps / MB,
                                        peak_svd_MB=pv / MB))
    # summaries: median over cells and chi_s
    arms = sorted({r['arm'] for r in out['rows']})
    for arm in arms:
        for N in Ns:
            sel = [r for r in out['rows'] if r['arm'] == arm and r['N'] == N]
            lines.append(f'{arm:20s} N={N:3d}: median peak-memory ratio spcf/SVD(chi_eq) {np.median([r["mem_ratio"] for r in sel]):.2f} '
                         f'(range {min(r["mem_ratio"] for r in sel):.2f}-{max(r["mem_ratio"] for r in sel):.2f}), time ratio {np.median([r["time_ratio"] for r in sel]):.2f} '
                         f'(range {min(r["time_ratio"] for r in sel):.2f}-{max(r["time_ratio"] for r in sel):.2f}), median chi_eq/chi_s {np.median([r["chi_eq"] / r["chi"] for r in sel]):.2f}, n={len(sel)}')
    json.dump(out, open(os.path.join(RES, 'pareto_projection.json'), 'w'), indent=1, default=float)
    open(os.path.join(RES, 'pareto_projection.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
