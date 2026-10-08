"""Tables, scaling exponents and figure for results/cost_chunk.json (cost_chunk.py).

    python cost_chunk_analysis.py   -> results/cost_chunk_tables.md, results/cost_chunk_fits.json, figures/cost_chunk.png
"""
import os
import json
import numpy as np
import _p2  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MB = 2 ** 20


def load():
    rows = json.load(open(os.path.join(RES, 'cost_chunk.json')))
    d = {}
    for r in rows:
        d.setdefault(r['variant'], {}).setdefault(r['chi'], {})[r['what']] = r
    # earlier measurement of the unchunked a = 2 at chi = 64 and a = 1 (cost_scaling_*_N16.json) for cross-reference only
    return d


def fit(chis, vals, lo=32):
    x = np.log([c for c, v in zip(chis, vals) if c >= lo and v and v > 0])
    y = np.log([v for c, v in zip(chis, vals) if c >= lo and v and v > 0])
    if len(x) < 3:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ sol
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1) / np.sum((x - x.mean()) ** 2))) if len(x) > 2 else float('nan')
    return float(sol[0]), se, len(x)


def main():
    d = load()
    chis = [16, 32, 48, 64, 96, 128]
    variants = [v for v in ['svd', 'a1', 'a1c1', 'a1c8', 'a1c64', 'a1c8f', 'a1c64f', 'a2', 'a2c1', 'a2c8', 'a2c64', 'a2c1f', 'a2c8f'] if v in d]
    figv = [v for v in ['svd', 'a1', 'a1c8f', 'a1c64f', 'a2', 'a2c1f', 'a2c8f'] if v in d]
    out, md = {}, []

    def val(v, chi, what):
        r = d.get(v, {}).get(chi, {}).get(what)
        if r is None:
            return None
        return r['t_med'] if what == 'time' else r['peak']
    md.append('Per-cut time (ms), median of the repeats, N = 16 bulk bond, float32, options of Tests 1/2, one BLAS thread\n')
    md.append('| variant | ' + ' | '.join(f'chi={c}' for c in chis) + ' | exponent (chi >= 32) |')
    md.append('|---|' + '---|' * (len(chis) + 1))
    for v in variants:
        vals = [val(v, c, 'time') for c in chis]
        f = fit(chis, vals)
        out[f'time|{v}'] = f
        md.append(f'| {v} | ' + ' | '.join('-' if x is None else f'{x * 1e3:.1f}' for x in vals) + f' | {"-" if f is None else f"{f[0]:.2f} +- {f[1]:.2f}"} |')
    md.append('\nTime ratio to the SVD cut\n')
    md.append('| variant | ' + ' | '.join(f'chi={c}' for c in chis) + ' |')
    md.append('|---|' + '---|' * len(chis))
    for v in variants[1:]:
        md.append(f'| {v} | ' + ' | '.join('-' if val(v, c, "time") is None or val('svd', c, "time") is None else f'{val(v, c, "time") / val("svd", c, "time"):.1f}' for c in chis) + ' |')
    for what, name in (('rss', 'Transient peak memory of one cut (MB), VmHWM growth, inputs and F tables excluded'), ('tmem', 'Transient peak memory of one cut (MB), tracemalloc')):
        md.append(f'\n{name}\n')
        md.append('| variant | ' + ' | '.join(f'chi={c}' for c in chis) + ' | exponent (chi >= 32) |')
        md.append('|---|' + '---|' * (len(chis) + 1))
        for v in variants:
            vals = [val(v, c, what) for c in chis]
            f = fit(chis, vals)
            out[f'{what}|{v}'] = f
            md.append(f'| {v} | ' + ' | '.join('-' if x is None else f'{x / MB:.1f}' for x in vals) + f' | {"-" if f is None else f"{f[0]:.2f} +- {f[1]:.2f}"} |')
        md.append('\nMemory ratio to the SVD cut\n')
        md.append('| variant | ' + ' | '.join(f'chi={c}' for c in chis) + ' |')
        md.append('|---|' + '---|' * len(chis))
        for v in variants[1:]:
            md.append(f'| {v} | ' + ' | '.join('-' if val(v, c, what) is None or val('svd', c, what) is None else f'{val(v, c, what) / max(val("svd", c, what), 1):.1f}' for c in chis) + ' |')
    tb = []
    for v in variants:
        r = d[v].get(chis[-1], {}).get('tmem') or d[v].get(chis[0], {}).get('tmem')
        if r and r.get('tables_bytes') is not None:
            tb.append(f'{v}: {r["tables_bytes"] / MB:.1f} MB')
    md.append('\nState independent F tables held by the cut (persistent, not in the transient numbers above): ' + ', '.join(tb))
    open(os.path.join(RES, 'cost_chunk_tables.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(out, open(os.path.join(RES, 'cost_chunk_fits.json'), 'w'), indent=1)
    print('\n'.join(md))

    INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
    col = {'svd': '#2a78d6', 'a1': '#1baf7a', 'a2': '#eb6834'}
    mk = {'': 'x', 'c1': 'o', 'c8': 's', 'c64': '^'}   # chunked variants in the figure use w0_fast (suffix f)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, what, yl, sc in ((axs[0], 'time', 'time per cut (s)', 1.0), (axs[1], 'rss', 'transient peak memory per cut (MB)', MB)):
        for v in figv:
            fam = v[:2] if v != 'svd' else 'svd'
            suffix = v[2:].rstrip('f') if v != 'svd' else ''
            xs = [c for c in chis if val(v, c, what) is not None]
            if not xs:
                continue
            ys = [val(v, c, what) / sc for c in xs]
            ax.plot(xs, ys, color=col[fam], marker=mk[suffix] if v != 'svd' else 'D', ms=5.5, lw=1.4, ls='--' if suffix == '' and v != 'svd' else '-',
                    mfc='white' if suffix == '' and v != 'svd' else col[fam], label='SVD cut' if v == 'svd' else f'spcf a = {v[1]}, ' + ('unchunked' if suffix == '' else f'chunked, block {suffix[1:]}'))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(chis)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticklabels([str(c) for c in chis])
        ax.set_xlabel('chi (N = 16, bulk bond)', fontsize=9, color=INK)
        ax.set_ylabel(yl, fontsize=9, color=INK)
        ax.grid(True, which='major', color=GRID, lw=0.7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axs[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    axs[0].set_title('Time of one fired cut', fontsize=10, loc='left', color=INK)
    axs[1].set_title('Peak memory inside one cut (resident-set high-water growth)', fontsize=10, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figures', 'cost_chunk.png'), dpi=140)


if __name__ == '__main__':
    main()
