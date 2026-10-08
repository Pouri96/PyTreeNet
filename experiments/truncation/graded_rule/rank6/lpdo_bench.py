"""Cell runner for the rank-6 first move: LPDO-TEBD with dephasing noise against the dense Lindblad reference.

    from lpdo_bench import cell
    row = cell('ising', 8, 4.0, 0.1, gamma, chi, kappa, probe_a=(2, 1), mode='mps')    # LPDO-TEBD, both truncations (chi, kappa)
    row = cell(..., mode='kraus')                                                      # chi = infinity, dense global Kraus truncation only
Rows are plain dicts (JSON-serialisable).  Reference density matrices are cached in _refcache/.
"""
import _p6  # noqa: F401
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import mpsenh as M
import lpdo as L
import gatelog as G

HERE = Path(__file__).resolve().parent
REF = HERE / '_refcache'
REF.mkdir(exist_ok=True)


def reference(model, N, T, dt, gamma):
    fn = REF / f'lindblad_{model}_N{N}_T{T:g}_dt{dt:g}_g{gamma:g}.npy'
    if fn.exists():
        return np.load(fn)
    rho = L.dense_lindblad(model, N, int(round(T / dt)), dt, gamma)
    np.save(fn, rho)
    return rho


_OBS = {}


def ref_obs(model, N, T, dt, gamma):
    key = (model, N, T, dt, gamma)
    if key not in _OBS:
        _OBS[key] = L.local_obs_rho(reference(model, N, T, dt, gamma), N, model)
    return _OBS[key]


def gate_summary(probe, a, ncuts_total):
    lg = np.array(probe.log[a]) if probe.log[a] else np.zeros((0, 4))
    if len(lg) == 0:
        return dict(a=a, n_logged=0, n_fire=0, on_frac_all=0.0, on_frac_logged=0.0, med_ratio=float('nan'), ncuts=ncuts_total)
    fire = lg[:, 1] >= lg[:, 2]
    ratio = lg[:, 1] / lg[:, 2]
    return dict(a=a, n_logged=int(len(lg)), n_fire=int(fire.sum()), on_frac_all=float(fire.sum() / ncuts_total), on_frac_logged=float(fire.mean()),
                med_ratio=float(np.median(ratio)), med_Bres=float(np.median(lg[:, 1])), med_tail=float(np.median(lg[:, 0])), ncuts=ncuts_total)


def cell(model, N, T, dt, gamma, chi, kappa, probe_a=None, mode='mps', keep_log=False):
    nsteps = int(round(T / dt))
    rho_ex = reference(model, N, T, dt, gamma)
    obs_ex = ref_obs(model, N, T, dt, gamma)
    row = dict(model=model, N=N, T=T, dt=dt, gamma=gamma, chi=chi, kappa=kappa, mode=mode)
    if mode == 'mps':
        pr = G.GateProbe(model, N, a=tuple(probe_a)) if probe_a else None
        Tl, info = L.run_lpdo(model, N, chi, kappa, nsteps, dt, gamma, probe=pr)
        rho = L.lpdo_to_rho(Tl)
        row.update({k: v for k, v in info.items()})
        if pr is not None:
            ncuts = nsteps * 2 * (N - 1)
            row['gate'] = {str(a): gate_summary(pr, a, ncuts) for a in probe_a}
            row['n_trunc_cuts'] = pr.ntrunc
            if keep_log:
                row['gate_log'] = {str(a): [list(map(float, e)) for e in pr.log[a]] for a in probe_a}
                row['gate_entries'] = [[int(e[0]), int(e[1]), e[2], float(e[3])] for e in pr.entries]
    elif mode == 'kraus':
        rho, info = L.dense_kraus_only(model, N, nsteps, dt, gamma, kappa)
        row.update(info)
    else:
        raise ValueError(mode)
    row.update(L.errors_rho(rho_ex, rho, N, model, obs_ex))
    return row


def _tojson(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dump(rows, fn):
    json.dump(rows, open(fn, 'w'), default=_tojson, indent=0)
