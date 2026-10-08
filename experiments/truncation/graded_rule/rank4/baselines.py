"""Matched-memory baselines for the Heisenberg-MPO experiments.

pp_traj   : coefficient-threshold Pauli propagation (pauli_prop.propagate's keep step), but the value is recorded after
            every Strang step (the sequence is a palindrome, so one run gives all n).  Optional xSPD X-weight cap
            (Begusic-Chan: drop strings with more than M X/Y factors) and optional pi_m^{sigma0} folding.
mps_traj  : Schroedinger MPS-TEBD (mpsenh.svd_cut, same gates and order as mpsenh.run_tebd) recording <Z_i0> every step.
Stored numbers: PP = 2 per string (int64 key + float coefficient); MPS = number of tensor elements (complex counted as
one, the repo convention of mpsenh.stored_params); MPO = real elements.
"""
import sys
import time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import _paths  # noqa: F401,E402
import mpsenh as M  # noqa: E402
import pauli_prop as PP  # noqa: E402
import heis_mpo as H  # noqa: E402

_XYMASK = np.int64(0x5555555555555555)


def _nxy(keys):
    return np.bitwise_count((keys ^ (keys >> 1)) & _XYMASK)


def _eval(keys, coef, v):
    """sum_P c_P prod_i v_i[p_i]."""
    N = v.shape[0]
    val = coef.copy()
    for s in range(N):
        d = (keys >> (2 * s)) & 3
        val *= v[s][d]
    return float(val.sum())


def pp_traj(model, N, nsteps, dt, i0, pauli, eps, xcap=None, fold=None, cap=3_000_000, tmax=1500):
    """Returns dict(val[n], peak_strings, params, wall, truncated, last_n)."""
    Gs = M.make_gates(model, N, dt)
    Rs = [PP.transfer(G.reshape(4, 4)) for G in Gs]
    v = H.initial_covectors(model, N)
    keys = np.array([np.int64(pauli) << (2 * i0)])
    coef = np.array([1.0])
    one = list(range(N - 1)) + list(range(N - 2, -1, -1))
    val = [_eval(keys, coef, v)]
    peak = 1
    t0 = time.time()
    trunc = False
    for n in range(1, nsteps + 1):
        for b in one:
            R = Rs[b]
            sh = 2 * b
            code = (keys >> sh) & 15
            touched = code != 0
            if not touched.any():
                continue
            ku, cu = keys[~touched], coef[~touched]
            kt, ct, codet = keys[touched], coef[touched], code[touched]
            base = kt & ~np.int64(15 << sh)
            outk, outc = [ku], [cu]
            for j in range(16):
                r = R[codet, j]
                m = np.abs(r) > 1e-14
                if m.any():
                    nk = base[m] | np.int64(j << sh)
                    nc = ct[m] * r[m]
                    if xcap is not None:
                        okx = _nxy(nk) <= xcap
                        nk, nc = nk[okx], nc[okx]
                    outk.append(nk)
                    outc.append(nc)
            nk, nc = np.concatenate(outk), np.concatenate(outc)
            uk, inv = np.unique(nk, return_inverse=True)
            uc = np.bincount(inv, weights=nc, minlength=len(uk))
            keep = np.abs(uc) >= eps
            keys, coef = uk[keep], uc[keep]
            peak = max(peak, len(keys))
            if len(keys) > cap or time.time() - t0 > tmax:
                trunc = True
                break
        if trunc:
            break
        val.append(_eval(keys, coef, v))
    last = len(val) - 1
    val = np.array(val + [np.nan] * (nsteps + 1 - len(val)))
    return dict(val=val, peak_strings=int(peak), params=int(2 * peak), wall=time.time() - t0, truncated=trunc, last_n=last)


def _meas(T, N, i0, Op):
    env = np.ones((1, 1), dtype=complex)
    nrm = np.ones((1, 1), dtype=complex)
    for i, t in enumerate(T):
        if i == i0:
            env = np.einsum('ab,asc,st,btd->cd', env, t.conj(), Op, t)
        else:
            env = np.einsum('ab,asc,bsd->cd', env, t.conj(), t)
        nrm = np.einsum('ab,asc,bsd->cd', nrm, t.conj(), t)
    return float((env[0, 0] / nrm[0, 0]).real)


def mps_traj(model, N, chi, nsteps, dt, i0):
    Gs = M.make_gates(model, N, dt)
    T = M.initial_mps(model, N)
    val = [_meas(T, N, i0, M.Z)]
    params = [M.stored_params(T)]
    t0 = time.time()
    for n in range(1, nsteps + 1):
        for b in range(N - 1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            T[b], T[b + 1], _, _ = M.svd_cut(th, chi, 'R')
        for b in range(N - 2, -1, -1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            T[b], T[b + 1], _, _ = M.svd_cut(th, chi, 'L')
        val.append(_meas(T, N, i0, M.Z))
        params.append(M.stored_params(T))
    return dict(val=np.array(val), params=int(max(params)), wall=time.time() - t0, maxbond=max(M.ranks(T)))
