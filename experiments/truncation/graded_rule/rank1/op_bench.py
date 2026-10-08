"""Operator-TEBD bench: arms x chi ladder -> JSON, with the pre-registered metrics split into three regions by the exact C_Z.

    python rank1/op_bench.py MODEL N T dt ARMS CHIS OUT [nproc] [diag]
    arms: svd | rw:G | spcop[:key=val;key=val...]      (svdraw is derived from the svd run: same trajectory times the norm^2 scalar)
Reference (dense Heisenberg circuit with the same gates) is cached in rank1/results/refcache/.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import op_tebd as O
import spcop

CACHE = HERE / 'results' / 'refcache'
SNAP_EVERY = 10                      # dense snapshots every 10 steps (t = 1.0 at dt = 0.1)
THETAS = (0.1, 0.5)
REG_EDGES = (1e-2, 0.5)              # ahead < 1e-2 <= front <= 0.5 < behind (by the EXACT C_Z)


# ------------------------------------------------------------------ reference
def get_ref(model, N, T, dt):
    nsteps = int(round(T / dt))
    fn = CACHE / f'ref_{model}_N{N}_T{T:g}_dt{dt:g}.npz'
    if fn.exists():
        z = np.load(fn)
        snaps = {int(k[2:]): z[k] for k in z.files if k.startswith('c_')}
        return dict(p1=z['p1'], cz0=z['cz0'], snaps=snaps, nsteps=nsteps, dt=dt, N=N, model=model)
    t0 = time.time()
    snap_steps = tuple(range(SNAP_EVERY, nsteps + 1, SNAP_EVERY))
    p1, cz0, snaps = O.dense_series(model, N, nsteps, dt, snap_steps=snap_steps)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(fn, p1=p1, cz0=cz0, **{f'c_{k}': v for k, v in snaps.items()})
    print(f'reference {fn.name} built in {time.time() - t0:.0f}s', flush=True)
    return dict(p1=p1, cz0=cz0, snaps=snaps, nsteps=nsteps, dt=dt, N=N, model=model)


# ------------------------------------------------------------------ pair (joint) marginals of a normalised dense c
def pair_marginals(c, N):
    """dict d -> array (N-d, 16) of p_{x,x+d}(a,b), for d = 1 .. N-1."""
    c2 = (c * c).reshape([4] * N)
    out = {}
    # sum out leading axes progressively: fine for N = 10 (a few seconds)
    for x in range(N - 1):
        pass
    for d in range(1, N):
        rows = []
        for x in range(N - d):
            other = tuple(j for j in range(N) if j not in (x, x + d))
            rows.append(c2.sum(axis=other).reshape(-1))
        out[d] = np.array(rows)
    return out


def window_marginals(c, N, k):
    """array (N-k+1, 4^k) of the k-site consecutive-window label marginals."""
    c2 = (c * c).reshape([4] * N)
    rows = []
    for x in range(N - k + 1):
        other = tuple(j for j in range(N) if not (x <= j < x + k))
        rows.append(c2.sum(axis=other).reshape(-1))
    return np.array(rows)


ODD2 = np.array([(1 if (a in (1, 2)) ^ (b in (1, 2)) else 0) for a in range(4) for b in range(4)], dtype=float)   # {X,Y} anticommute with Z


def snapshot_stats(c, cex, N):
    """Metrics of the dense normalised state c against the exact cex (both flat, normalised)."""
    out = {}
    out['infid'] = float(1.0 - (c @ cex) ** 2)
    for k in (1, 2, 3):
        a, b = window_marginals(c, N, k), window_marginals(cex, N, k)
        out[f'win{k}_rms'] = float(np.sqrt(np.mean((a - b) ** 2)))
    pa, pb = pair_marginals(c, N), pair_marginals(cex, N)
    for d in range(1, N):
        out[f'pair_d{d}_rms'] = float(np.sqrt(np.mean((pa[d] - pb[d]) ** 2)))
    far = np.concatenate([(pa[d] - pb[d]).ravel() for d in range(3, N)])
    out['far_ge3_rms'] = float(np.sqrt(np.mean(far ** 2)))
    out['ZZ_otoc_d3_rms'] = float(np.sqrt(np.mean((2 * (pa[3] @ ODD2) - 2 * (pb[3] @ ODD2)) ** 2)))
    out['ZZ_otoc_d3_max'] = float(np.max(np.abs(2 * (pa[3] @ ODD2) - 2 * (pb[3] @ ODD2))))
    return out


# ------------------------------------------------------------------ one run
def make_cut(arm, N, diag):
    """returns (cut, gamma)"""
    if arm == 'svd':
        return spcop.OpCut(N, diag=diag), 1.0
    if arm.startswith('rw:'):
        return O.svd_cut_op, float(arm.split(':')[1])
    if arm.startswith('spcop'):
        kw = {}
        if ':' in arm:
            for item in arm.split(':', 1)[1].split(';'):
                if not item:
                    continue
                key, val = item.split('=')
                if key == 'ks':
                    kw['ks'] = tuple(int(ch) for ch in val)
                elif key in ('iters', 'passes'):
                    kw[key] = int(val)
                elif key == 'a':
                    kw['aL'] = kw['aR'] = int(val)
                else:
                    kw[key] = float(val)
        return spcop.SPCOp(N, **kw), 1.0
    raise ValueError(arm)


def run_arm(model, N, chi, T, dt, arm, diag=True, site=0):
    ref = get_ref(model, N, T, dt)
    nsteps = ref['nsteps']
    cut, gamma = make_cut(arm, N, diag)
    gops = O.pauli_gates(model, N, dt, gamma=gamma)
    p1 = np.zeros((nsteps + 1, N, 4))
    cz0 = np.zeros(nsteps + 1)
    lk = np.zeros(nsteps + 1)
    p1[0] = ref['p1'][0]
    cz0[0] = ref['cz0'][0]
    idx0 = 3 * 4 ** (N - 1 - site)
    stats = {}
    ncuts = []

    def snap(step, T_):
        c = O.phys_dense(T_, gamma)
        p1[step] = O.marginal1(c, N)
        cz0[step] = c[idx0]
        lk[step] = getattr(cut, 'logkept', 0.0)
        if step in ref['snaps']:
            cex = ref['snaps'][step]
            stats[step] = snapshot_stats(c, cex / np.linalg.norm(cex), N)
        ncuts.append(len(getattr(cut, 'log', [])))

    Tm, wall = O.run_op_tebd(N, chi, nsteps, gops, cut, snap=snap, site=site)
    rec = dict(model=model, N=N, chi=chi, T=T, dt=dt, arm=arm, wall=wall, maxrank=int(max(O.ranks(Tm))),
               p1=p1.tolist(), cz0=cz0.tolist(), logkept=lk.tolist(), snaps={str(k): v for k, v in stats.items()},
               ncuts_by_step=ncuts)
    if hasattr(cut, 'log') and len(cut.log):
        rec['cutlog'] = np.array(cut.log).tolist()
        rec['cutlog_cols'] = spcop.LOG_COLS
    if isinstance(cut, spcop.SPCOp):
        rec.update(fired=cut.fired, rejected=cut.rejected, skipped=cut.skipped, trunc=cut.trunc, calls=cut.calls,
                   tm={k: round(v, 3) for k, v in cut.tm.items()}, achosen={str(k): v for k, v in cut.achosen.items()})
    elif isinstance(cut, spcop.OpCut):
        rec.update(trunc=cut.trunc, calls=cut.calls, tm={k: round(v, 3) for k, v in cut.tm.items()})
    return rec


# ------------------------------------------------------------------ metrics
def first_cross(t, v, theta):
    """first time the series v(t) reaches theta, linear interpolation; nan if never."""
    idx = np.where(v >= theta)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (theta - v[i - 1]) * (t[i] - t[i - 1]) / (v[i] - v[i - 1]))


def region_stats(err, Cex):
    """err, Cex of equal shape; returns dict region -> (rms, max, mean_signed, n)."""
    out = {}
    masks = dict(ahead=Cex < REG_EDGES[0], front=(Cex >= REG_EDGES[0]) & (Cex <= REG_EDGES[1]), behind=Cex > REG_EDGES[1])
    masks['all'] = np.ones_like(Cex, dtype=bool)
    for name, m in masks.items():
        if m.any():
            e = err[m]
            out[name] = dict(rms=float(np.sqrt(np.mean(e ** 2))), max=float(np.max(np.abs(e))), mean=float(np.mean(e)), n=int(m.sum()))
        else:
            out[name] = dict(rms=float('nan'), max=float('nan'), mean=float('nan'), n=0)
    return out


def contour_lags(p1a, p1e, dt, kind='Z'):
    """lag Delta t(theta) = t_approx - t_exact of the first crossing of C(x, t) = theta, per site x >= 1 that the exact one reaches."""
    n1 = p1e.shape[0]
    t = np.arange(n1) * dt
    Ca, Ce = O.C_from_p1(p1a, kind), O.C_from_p1(p1e, kind)
    res = {}
    for th in THETAS:
        lags, cens = [], 0
        for x in range(p1e.shape[1]):
            te = first_cross(t, Ce[:, x], th)
            if np.isnan(te):
                continue
            ta = first_cross(t, Ca[:, x], th)
            if np.isnan(ta):
                lags.append(t[-1] - te)           # lower bound: never reached within T
                cens += 1
            else:
                lags.append(ta - te)
        lags = np.array(lags)
        res[str(th)] = dict(max=float(np.max(lags)) if len(lags) else float('nan'), mean=float(np.mean(lags)) if len(lags) else float('nan'),
                            min=float(np.min(lags)) if len(lags) else float('nan'), n=len(lags), censored=cens)
    return res


def metrics(rec, ref, raw=False):
    """rec: run record.  raw=True scales the one-site marginals by the cumulative kept weight (the unrenormalised 'svdraw' operator)."""
    p1e = ref['p1']
    p1a = np.array(rec['p1'])
    scale = np.exp(np.array(rec['logkept']))[:, None, None] if raw else 1.0
    p1a = p1a * scale
    dt = ref['dt']
    out = dict(model=rec['model'], N=rec['N'], chi=rec['chi'], arm=('svdraw' if raw else rec['arm']), wall=rec['wall'])
    Cz_e, Cz_a = O.C_from_p1(p1e[1:]), O.C_from_p1(p1a[1:])
    w_e = O.weight_from_p1(p1e[1:])
    w_a = (1.0 - p1a[1:, :, 0]) if not raw else (p1a[1:, :, 1:].sum(axis=-1))
    out['dC_Z'] = region_stats(Cz_a - Cz_e, Cz_e)
    out['dC_X'] = region_stats(O.C_from_p1(p1a[1:], 'X') - O.C_from_p1(p1e[1:], 'X'), Cz_e)
    out['dw'] = region_stats(w_a - w_e, Cz_e)
    out['lag_Z'] = contour_lags(p1a, p1e, dt)
    out['dp1_rms_T'] = float(np.sqrt(np.mean((p1a[-1] - p1e[-1]) ** 2)))
    out['cz0_rms'] = float(np.sqrt(np.mean((np.array(rec['cz0']) - ref['cz0']) ** 2)))
    out['cz0_rms_T'] = float(abs(rec['cz0'][-1] - ref['cz0'][-1]))
    if raw:
        out['norm2_T'] = float(np.exp(rec['logkept'][-1]))
        out['norm2_min_C_sat'] = float(np.exp(rec['logkept'][-1]))
    else:
        snaps = rec['snaps']
        last = str(max(int(k) for k in snaps)) if snaps else None
        if last is not None:
            out['final'] = snaps[last]
            out['snap_infid'] = {k: v['infid'] for k, v in snaps.items()}
    # local-vs-propagated: rms span-2 residual of the cuts of the last 5 steps / final rms one-site marginal error
    if 'cutlog' in rec:
        lg = np.array(rec['cutlog'])
        ncs = rec['ncuts_by_step']
        nsteps = len(ncs)
        lo = ncs[max(nsteps - 6, 0)] if nsteps > 5 else 0
        sel = lg[lo:]
        if len(sel):
            out['cut_span2_rms_last5'] = float(np.sqrt(np.mean(sel[:, 8] ** 2)))
            out['cut_rms_all_last5'] = float(np.sqrt(np.mean(sel[:, 4] ** 2)))
            out['ratio_cut_to_final'] = out['cut_span2_rms_last5'] / max(out['dp1_rms_T'], 1e-300)
        out['n_trunc_cuts'] = len(lg)
        out['cut_span2_rms_all'] = float(np.sqrt(np.mean(lg[:, 8] ** 2)))
    return out


# ------------------------------------------------------------------ CLI
def _job(a):
    return run_arm(*a)


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms = sys.argv[5].split(',')
    chis = [int(c) for c in sys.argv[6].split(',')]
    out = sys.argv[7]
    nproc = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    diag = bool(int(sys.argv[9])) if len(sys.argv) > 9 else True
    get_ref(model, N, T, dt)
    jobs = [(model, N, c, T, dt, a, diag) for c in chis for a in arms]
    res = []
    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(_job, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"done {r['model']} chi={r['chi']} {r['arm']} wall={r['wall']:.1f}s ({len(res)}/{len(jobs)}, {time.time() - t0:.0f}s)", flush=True)
