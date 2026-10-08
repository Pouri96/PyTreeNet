"""Pareto analysis at the final state (T = 4): final rdm2 against PEAK memory over the run and against total CPU time, per cell.

    python pareto_analysis.py      -> results/pareto_points.json, results/pareto_front.json, results/pareto_front.txt,
                                      figures/pareto_peak_mem.png, figures/pareto_peak_time.png

Data
  error, CPU time, stored parameters: results/eqerr_raw_<cell>.json (SVD plain / back, DMT, spcf a = 2 unchunked: existing runs, not redone),
                                      results/a1_traj.json and results/pareto_time.json (new: spcf a = 1, chunked variants)
  peak memory: results/pareto_rss.json  (VmHWM growth over the whole run, fresh process per run, see traj.py)  [primary]
               results/pareto_tmem.json (tracemalloc peak, decomposition; cross-check)
Front rule (rank2_practicality.md, P3): a spcf point is ON the front for a cost axis iff no baseline point (SVD plain, SVD back, DMT) has cost <= and
error <= its own.  The equal-error factor = cost of the spcf point / cost of the baseline lower envelope at the spcf point's error (log-log interpolation
along the non-dominated baseline points).
"""
import os
import json
import numpy as np
import _p2  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
CELLS = ['stag_mu1', 'stag_mu0.5', 'stag_mu0.25', 'dw_mu1', 'dw_mu0.5', 'dw_mu0.25']
TITLE = {'stag': 'staggered', 'dw': 'domain wall'}
BASE = ['svd_plain', 'svd_back', 'dmt']
MB = 2 ** 20


def load_points(cell):
    pts = {}

    def put(arm, chi, **kw):
        pts.setdefault((arm, chi), {}).update(kw)
    d = json.load(open(os.path.join(RES, f'eqerr_raw_{cell}.json')))
    cpus = {}
    for r in d['rows']:
        arm = 'spcf-a2' if r['arm'] == 'spcf' else r['arm']
        s = r['series'][-1]
        if r['rep'] == 0:
            put(arm, r['chi'], err=s['rdm2'], nn=s['nn_rms'], cpu=s['cpu'], params=s['params'], src='eqerr_raw', fired=s.get('fired'), calls=s.get('calls'))
    for fn in ('a1_traj.json', 'pareto_time.json', 'pareto_time_gplain.json'):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            for r in json.load(open(p)):
                if r['cell'] == cell and r.get('mode', 'time') == 'time' and r['arm'] not in ('spcf-a2',):
                    s = r['series'][-1]
                    put(r['arm'], r['chi'], err=s['rdm2'], nn=s['nn_rms'], cpu=s['cpu'], params=s['params'], src=fn, fired=s.get('fired'), calls=s.get('calls'))
    for fn, key in (('pareto_rss.json', 'rssB'), ('pareto_rss_spcf.json', 'rssB'), ('pareto_rss_gplain.json', 'rssB'), ('pareto_rssA.json', 'rssA'), ('pareto_tmem.json', 'tmemB')):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            for r in json.load(open(p)):
                if r['cell'] == cell:
                    v = r['peak_delta'] if key.startswith('rss') else r['tm_peak']
                    put(r['arm'], r['chi'], **{key: v / MB, 'tables_MB': r.get('tables_bytes', 0) / MB})
                    if key == 'tmemB':
                        put(r['arm'], r['chi'], tm_mps=r['mps_bytes'] / MB)
    for p in pts.values():                              # headline peak memory = growth of the resident set with the tables resident but their build transient excluded
        if 'rssB' in p:
            p['rss'] = p['rssB'] + p.get('tables_MB', 0.0)
    return pts


def envelope(points):
    """non-dominated (cost, err) points (lower-left front) sorted by cost; points = [(cost, err, tag)]"""
    ps = sorted(points, key=lambda p: (p[0], p[1]))
    out, best = [], np.inf
    for c, e, t in ps:
        if e < best:
            out.append((c, e, t))
            best = e
    return out


def baseline_cost_at(env, e):
    """cost of the baseline lower envelope at error e: log-log interpolation along the envelope (err decreasing with cost); flag if outside."""
    c = np.array([p[0] for p in env])
    er = np.array([p[1] for p in env])
    if e >= er[0]:
        return c[0], 'below_range'          # even the cheapest baseline point is at or below this error: cost is an upper bound on what a baseline needs? (baseline cheaper)
    if e < er[-1]:
        return c[-1], 'above_range'         # the best baseline point is still worse: baseline cost is a LOWER bound (ratio is an upper bound)
    # er decreasing along the envelope
    i = int(np.argmax(er <= e))
    f = (np.log(e) - np.log(er[i - 1])) / (np.log(er[i]) - np.log(er[i - 1]))
    return float(np.exp(np.log(c[i - 1]) + f * (np.log(c[i]) - np.log(c[i - 1])))), 'ok'


def analyse(allpts, axis):
    """front membership and equal-error factors for every spcf variant, per cell; axis in ('rss', 'cpu')"""
    out = {}
    for cell, pts in allpts.items():
        base = [(p[axis], p['err'], (a, chi)) for (a, chi), p in pts.items() if a in BASE and axis in p]
        env = envelope(base)
        res = {}
        for (a, chi), p in sorted(pts.items()):
            if a in BASE or axis not in p:
                continue
            dom = [q for q in base if q[0] <= p[axis] and q[1] <= p['err']]
            fac, flag = baseline_cost_at(env, p['err'])
            res.setdefault(a, []).append(dict(chi=chi, cost=p[axis], err=p['err'], on_front=not dom, dominated_by=[d[2] for d in dom][:3], factor=p[axis] / fac,
                                              base_cost=fac, flag=flag))
        out[cell] = dict(variants=res, envelope=[(c, e, t) for c, e, t in env])
    return out


def style(ax):
    INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.7)
    ax.grid(True, which='minor', color=GRID, lw=0.3, alpha=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


# arm key -> (label, colour, marker, linestyle, filled)
ORANGE = '#eb6834'
STYLE = {'svd_plain': ('SVD, plain gauge', '#2a78d6', 'o', '-', True),
         'svd_back': ('SVD, backward gauge', '#1baf7a', 's', '-', True),
         'dmt': ('DMT', '#4a3aa7', '^', '-', True)}


def spcf_style(arm):
    kind = arm.split('-')[0]
    a = arm.split('-')[1]
    chunk = kind == 'spcfc'
    lab = f'spcf {a[0]} = {a[1:]}' + (', chunked' if chunk else ', unchunked') + (', plain gauge' if 'gplain' in arm else '')
    return (lab, ORANGE, 'D' if a == 'a2' else ('P' if 'gplain' in arm else 'o'), ('-.' if 'gplain' in arm else '-') if chunk else (0, (4, 2)), chunk)


def figure(allpts, axis, xlabel, fname, variants, title, fronts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED = '#1f1f1e', '#6b6a63'
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for ax, cell in zip(axs.ravel(), CELLS):
        style(ax)
        pts = allpts[cell]
        for arm in BASE + variants:
            xs = sorted([(chi, p) for (a, chi), p in pts.items() if a == arm and axis in p])
            if not xs:
                continue
            lab, col, mk, ls, filled = STYLE[arm] if arm in STYLE else spcf_style(arm)
            x = [p[axis] for _, p in xs]
            y = [p['err'] for _, p in xs]
            sp = arm not in STYLE
            ax.plot(x, y, ls=ls, color=col, lw=1.7 if sp else 1.4, marker=mk, ms=7 if sp else 4.5, mfc=col if filled else 'white', mec=col if not filled else 'white',
                    mew=1.4 if not filled else 1.0, label=lab, zorder=3 if sp else 2)
            if sp:
                for (chi, p), xx, yy in zip(xs, x, y):
                    ax.annotate(f'{chi}', (xx, yy), textcoords='offset points', xytext=(4, 4 if filled else -9), fontsize=7, color=INK)
        fam, mu = cell.split('_mu')
        ax.set_title(f'{TITLE[fam]}, mu = {mu}', fontsize=10, loc='left', color=INK)
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
        ax.set_ylabel('final rdm2 error (T = 4)', fontsize=8.5, color=INK)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(len(l), 7), frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(FIG, fname), dpi=140)
    plt.close(fig)


def main():
    allpts = {c: load_points(c) for c in CELLS}
    json.dump({c: {f'{a}|{chi}': p for (a, chi), p in pts.items()} for c, pts in allpts.items()}, open(os.path.join(RES, 'pareto_points.json'), 'w'), indent=1)
    arms = sorted({a for pts in allpts.values() for (a, _) in pts if a not in BASE})
    fronts = {ax: analyse(allpts, ax) for ax in ('rss', 'rssB', 'cpu')}
    json.dump(fronts, open(os.path.join(RES, 'pareto_front.json'), 'w'), indent=1, default=float)
    lines = []
    for ax, name in (('rss', 'PEAK MEMORY (VmHWM growth + resident F tables, MB)'), ('rssB', 'PEAK MEMORY without the resident F tables (VmHWM growth, MB)'), ('cpu', 'CPU TIME to T = 4 (s)')):
        lines.append(f'=== {name} ===')
        for arm in arms:
            on = 0
            lines.append(f'-- {arm}')
            fl = []
            for cell in CELLS:
                v = fronts[ax][cell]['variants'].get(arm)
                if not v:
                    continue
                anyf = any(x['on_front'] for x in v)
                on += anyf
                fac = [x['factor'] for x in v]
                lines.append(f'   {cell:12s} on front: {anyf!s:5s} points on front {[x["chi"] for x in v if x["on_front"]]}  equal-error cost factor per chi '
                             + ' '.join(f'{x["chi"]}:{x["factor"]:.2f}{"" if x["flag"] == "ok" else "(" + x["flag"][:3] + ")"}' for x in v))
                fl += fac
            lines.append(f'   cells with a point on the front: {on} of {len([c for c in CELLS if fronts[ax][c]["variants"].get(arm)])};  median factor over cells and chi {np.median(fl) if fl else float("nan"):.2f}')
            small = [x['factor'] for c in CELLS for x in fronts[ax][c]['variants'].get(arm, []) if x['chi'] <= 24]
            large = [x['factor'] for c in CELLS for x in fronts[ax][c]['variants'].get(arm, []) if x['chi'] >= 32]
            if small:
                lines.append(f'   chi <= 24: median {np.median(small):.2f} (min {min(small):.2f}, max {max(small):.2f}, n={len(small)})' +
                             (f';  chi >= 32: median {np.median(large):.2f} (min {min(large):.2f}, max {max(large):.2f}, n={len(large)})' if large else ''))
    open(os.path.join(RES, 'pareto_front.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    figure(allpts, 'rss', 'peak memory over the run (MB, resident-set high-water growth)', 'pareto_peak_mem.png',
           [a for a in arms if a != 'spcf-a2' or True],
           'Final-state Pareto (Ising N = 10, T = 4): final rdm2 error against PEAK memory over the whole run. Lower-left is better. Numbers on spcf points are chi.', fronts)
    figure(allpts, 'rssB', 'peak memory over the run without the resident F tables (MB)', 'pareto_peak_mem_notables.png', [a for a in arms],
           'Same as the memory figure with the constant F tables of spcf (about 8 MB at a = 2, 0.3 MB at a = 1, chunked) left out. Numbers on spcf points are chi.', fronts)
    figure(allpts, 'cpu', 'total CPU time to reach T = 4 (s)', 'pareto_peak_time.png', arms,
           'Final-state Pareto (Ising N = 10, T = 4): final rdm2 error against total CPU time. Lower-left is better. Numbers on spcf points are chi.', fronts)


if __name__ == '__main__':
    main()
