"""Projection of the N = 10 peak-memory comparison to longer chains (a MODEL anchored on the N = 10 measurements; labelled as such in the report).

    python pareto_projection.py    reads results/pareto_points.json, writes results/pareto_projection.json and results/pareto_projection.txt

For every chunked spcf point (variant V, chi_s) of a cell: chi_eq = bond dimension at which the better-gauge SVD reaches the same final rdm2 (log-log
interpolation of the monotone envelope, flagged if outside the ladder).  The measured whole-run peak at N = 10 (resident-set growth, F tables resident) is
the anchor; a chain with N sites has N - 10 more bulk sites, each stored as 4 chi^2 complex numbers (64 chi^2 bytes), and the transient inside a cut does
not depend on N for the SVD cut and for the chunked cuts (the window is local, the blocks are fixed):
        peak_N(V, chi_s) = peak_10(V, chi_s) + (N - 10) * 64 chi_s^2          peak_N(SVD, chi_eq) = peak_10(SVD, chi_eq) + (N - 10) * 64 chi_eq^2
The unchunked cuts are not projected (their transient grows with N until the outer bonds of the window reach chi; measured per cut in cost_chunk.json).
Time is not projected: the whole-run time ratio at equal chi is flat in chi (cost_chunk.json: the per-cut ratio is constant from chi = 16 to 128 and both
cuts scale with an exponent near 2.5), so the equal-error time factor is set by the accuracy gain alone; the break-even condition is stated in the report.
The assumption that chi_eq / chi_s measured at N = 10, T = 4 carries over is the weak point.
"""
import os
import json
import numpy as np
import _p2  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MB = 2 ** 20
CELLS = ['stag_mu1', 'stag_mu0.5', 'stag_mu0.25', 'dw_mu1', 'dw_mu0.5', 'dw_mu0.25']
VARIANTS = ['spcfc-a1-b8-wf', 'spcfc-a1-b8-wf-gplain', 'spcfc-a2-b1-wf']
NS = (10, 20, 50, 100, 200)


def main():
    pts = json.load(open(os.path.join(RES, 'pareto_points.json')))
    out, lines = [], []
    for cell in CELLS:
        P = {(k.split('|')[0], int(k.split('|')[1])): v for k, v in pts[cell].items()}
        chis = sorted({c for (a, c) in P if a == 'svd_plain' and ('svd_back', c) in P and 'rss' in P[('svd_plain', c)] and 'rss' in P[('svd_back', c)]})
        err = np.array([min(P[('svd_plain', c)]['err'], P[('svd_back', c)]['err']) for c in chis])
        env = np.minimum.accumulate(err)
        # memory of the SVD run that realises the envelope at each chi (the better gauge at that chi, carried along the envelope)
        mem = []
        for k, c in enumerate(chis):
            j = int(np.argmax(err[:k + 1] == env[k]))
            g = 'svd_plain' if P[('svd_plain', chis[j])]['err'] <= P[('svd_back', chis[j])]['err'] else 'svd_back'
            mem.append(P[(g, chis[j])]['rss'])
        mem = np.array(mem)
        for (arm, chi_s), p in sorted(P.items()):
            if arm not in VARIANTS or 'rss' not in p:
                continue
            e = p['err']
            if env[-1] > e:
                ce, flag = float(chis[-1]), 'above'
            elif env[0] <= e:
                ce, flag = float(chis[0]), 'below'
            else:
                i = int(np.argmax(env <= e))
                f = (np.log(e) - np.log(env[i - 1])) / (np.log(env[i]) - np.log(env[i - 1]))
                ce, flag = float(np.exp(np.log(chis[i - 1]) + f * (np.log(chis[i]) - np.log(chis[i - 1])))), 'ok'
            m10 = float(np.exp(np.interp(np.log(ce), np.log(chis), np.log(mem))))
            for N in NS:
                ps = p['rss'] + (N - 10) * 64 * chi_s ** 2 / MB
                pv = m10 + (N - 10) * 64 * ce ** 2 / MB
                out.append(dict(cell=cell, arm=arm, chi=chi_s, N=N, chi_eq=ce, flag=flag, peak_spcf_MB=ps, peak_svd_MB=pv, mem_ratio=ps / pv))
    arms = [a for a in VARIANTS if any(r['arm'] == a for r in out)]
    for arm in arms:
        for N in NS:
            sel = [r for r in out if r['arm'] == arm and r['N'] == N]
            lines.append(f'{arm:24s} N={N:3d}: peak-memory ratio spcf / SVD(chi_eq): median {np.median([r["mem_ratio"] for r in sel]):.2f}, range '
                         f'{min(r["mem_ratio"] for r in sel):.2f} to {max(r["mem_ratio"] for r in sel):.2f}, entries below 1: {sum(r["mem_ratio"] < 1 for r in sel)} of {len(sel)}; '
                         f'median chi_eq/chi_s {np.median([r["chi_eq"] / r["chi"] for r in sel]):.2f}')
    json.dump(out, open(os.path.join(RES, 'pareto_projection.json'), 'w'), indent=1, default=float)
    open(os.path.join(RES, 'pareto_projection.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
