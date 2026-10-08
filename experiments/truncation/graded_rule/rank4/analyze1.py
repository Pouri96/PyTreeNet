"""Test-1 analysis: tables against the pre-registered criteria + figure.  python analyze1.py RESULT.json [RESULT2.json ...]"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRE = [12, 16, 24, 32, 48]           # pre-registered chi set
ARMS = ['svd_raw', 'svd_renorm', 'dmt_I', 'dmt_neel']


def load(path):
    return json.load(open(path))


def mps_at_params(recs, params):
    """Log-log interpolation of the MPS dmax at equal ACTUAL stored numbers; None outside the ladder."""
    m = sorted([(r['params'], r['dmax'], r['chi']) for r in recs if r['kind'] == 'mps'])
    p = np.array([x[0] for x in m], float)
    d = np.array([x[1] for x in m], float)
    if params < p[0] or params > p[-1]:
        return None
    return float(np.exp(np.interp(np.log(params), np.log(p), np.log(d))))


def analyze(path, verbose=True):
    recs = load(path)
    model, N, T = recs[0]['model'], recs[0]['N'], recs[0]['T']
    mpo = {(r['arm'], r['chi']): r for r in recs if r['kind'] == 'mpo'}
    mps = {r['chi']: r for r in recs if r['kind'] == 'mps'}
    chis = sorted({c for (_, c) in mpo})
    lines = []
    P = lines.append
    P(f'### {model}, N={N}, T={T}, observable Z_{N // 2 - 1}\n')
    P('Delta_max over t in [0,T] (Delta(T) in brackets). chi = max bond of the MPO; `p` = stored real numbers.\n')
    P('| chi | p | SVD-raw | SVD-renorm | DMT-I | DMT-sigma0 | SVD-renorm/DMT-s0 | DMT-I/DMT-s0 | SVD-raw/DMT-s0 | pre-reg |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    ratios = {}
    for c in chis:
        row = {a: mpo[(a, c)] for a in ARMS if (a, c) in mpo}
        if len(row) < 4:
            continue
        r_sv = row['svd_renorm']['dmax'] / row['dmt_neel']['dmax']
        r_di = row['dmt_I']['dmax'] / row['dmt_neel']['dmax']
        r_sr = row['svd_raw']['dmax'] / row['dmt_neel']['dmax']
        ratios[c] = (r_sv, r_di, r_sr)
        cells = ' | '.join(f"{row[a]['dmax']:.2e} [{row[a]['dT']:.1e}]" for a in ARMS)
        P(f"| {c} | {row['svd_raw']['params']} | {cells} | {r_sv:.2f} | {r_di:.2f} | {r_sr:.2f} | {'yes' if c in PRE else 'suppl.'} |")
    # criteria
    P('')
    win = lambda c: 1e-3 <= mpo[('svd_renorm', c)]['dmax'] <= 1e-1
    P('Pre-registered success test: DMT-s0 >= 3x lower Delta_max than BOTH SVD-renorm and DMT-I at equal chi, for >= 2 chi in the '
      'window where SVD Delta_max is in [1e-3, 1e-1].')
    for label, cs in (('pre-registered chi set', [c for c in PRE if c in ratios]), ('all chi incl. supplementary', sorted(ratios))):
        ok = [c for c in cs if win(c) and ratios[c][0] >= 3 and ratios[c][1] >= 3]
        inwin = [c for c in cs if win(c)]
        P(f'- {label}: chi in window: {inwin}; chi meeting 3x vs both: {ok}  ->  {"SUCCESS-criterion-1 met" if len(ok) >= 2 else "NOT met"}')
    for label, cs in (('pre-registered chi set', [c for c in PRE if c in ratios]), ('all chi', sorted(ratios))):
        gain_min = {c: min(ratios[c][0], ratios[c][1]) for c in cs}
        gain_max = {c: max(ratios[c][0], ratios[c][1]) for c in cs}
        kill_either = all(g < 1.5 for g in gain_min.values())      # within 1.5x of at least one comparator at every chi
        kill_both = all(g < 1.5 for g in gain_max.values())        # within 1.5x of both at every chi
        P(f'- kill (reference adds nothing), {label}: within 1.5x of at least one of DMT-I/SVD-renorm at every chi: {kill_either}; '
          f'of both at every chi: {kill_both}')
    # memory-matched MPS
    P('\nMatched-memory MPS-TEBD (Schroedinger). Nominal rule chi_s = round(sqrt2 chi) (same element count 2 chi_s^2 = 4 chi^2, complex '
      'counted as one number); also interpolated at equal ACTUAL stored numbers.\n')
    P('| chi | best MPO arm | best MPO Delta_max | MPS chi_s (nominal) | MPS Delta_max | MPS actual p | MPS interp. at MPO p | best-MPO / MPS (nominal) | best-MPO / MPS (interp) |')
    P('|---|---|---|---|---|---|---|---|---|')
    mps_ratios = []
    for c in chis:
        if not all((a, c) in mpo for a in ARMS):
            continue
        best = min(ARMS, key=lambda a: mpo[(a, c)]['dmax'])
        bd = mpo[(best, c)]['dmax']
        cs = int(round(np.sqrt(2) * c))
        cs = min(mps, key=lambda k: abs(k - cs))
        md = mps[cs]['dmax']
        mi = mps_at_params(recs, mpo[(best, c)]['params'])
        mps_ratios.append((c, bd / md, (bd / mi) if mi else None))
        P(f"| {c} | {best} | {bd:.2e} | {cs} | {md:.2e} | {mps[cs]['params']} | {('%.2e' % mi) if mi else 'n/a'} | {bd / md:.2f} | {('%.2f' % (bd / mi)) if mi else 'n/a'} |")
    # PP
    pp = sorted([r for r in recs if r['kind'] == 'pp'], key=lambda r: r['params'])
    if pp:
        P('\nPauli propagation (eps ladder): memory = 2 numbers per string at the peak.\n')
        P('| eps | peak strings | p | Delta_max | Delta(T) | wall s | truncated |')
        P('|---|---|---|---|---|---|---|')
        for r in pp:
            P(f"| {r['eps']:g} | {r['peak_strings']} | {r['params']} | {r['dmax']:.2e} | {r['dT']:.2e} | {r['wall']:.0f} | {r['truncated']} |")
    P('\nMPS-TEBD ladder:\n')
    P('| chi_s | actual p | Delta_max | Delta(T) |')
    P('|---|---|---|---|')
    for k in sorted(mps):
        P(f"| {k} | {mps[k]['params']} | {mps[k]['dmax']:.2e} | {mps[k]['dT']:.2e} |")
    txt = '\n'.join(lines)
    if verbose:
        print(txt)
    return txt, recs


def figure(paths, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.2), squeeze=False)
    sty = {'svd_raw': ('C0', 'o'), 'svd_renorm': ('C9', 's'), 'dmt_I': ('C1', '^'), 'dmt_neel': ('C3', 'v')}
    for ax, p in zip(axs[0], paths):
        recs = load(p)
        for a in ARMS:
            rr = sorted([r for r in recs if r['kind'] == 'mpo' and r['arm'] == a], key=lambda r: r['params'])
            ax.plot([r['params'] for r in rr], [r['dmax'] for r in rr], color=sty[a][0], marker=sty[a][1], label=a, lw=1)
        rr = sorted([r for r in recs if r['kind'] == 'mps'], key=lambda r: r['params'])
        ax.plot([r['params'] for r in rr], [r['dmax'] for r in rr], 'k-x', label='MPS-TEBD', lw=1)
        rr = sorted([r for r in recs if r['kind'] == 'pp' and r['arm'] == 'pp'], key=lambda r: r['params'])
        ax.plot([r['params'] for r in rr], [r['dmax'] for r in rr], color='gray', marker='+', ls='--', label='Pauli prop.', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('stored numbers'); ax.set_ylabel(r'$\Delta_{max}$')
        ax.set_title(f"{recs[0]['model']} N={recs[0]['N']} T={recs[0]['T']}", fontsize=10)
        ax.grid(alpha=.3)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == '__main__':
    texts = []
    for p in sys.argv[1:]:
        t, _ = analyze(p)
        texts.append(t)
        print()
