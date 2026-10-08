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


def fold_strings(keys, coef, m, N, v):
    """pi_m^{sigma0} (Perez-Wojtowicz-Plenio 2609.12840 Eq. 4) for a product reference with single-site <Z> = v[i,3]=+-1,
    <X>=<Y>=0 (the Neel state): a string with weight > m is replaced by
        sum_{U subset of its Z-sites, |U| <= m - n_xy}  c * prod_{i in Z-sites \\ U} mu_i * S(n_rest, K) * (XY part x Z_U),
    S = sum_{j=0}^{K} (-1)^j C(n_rest, j), K = m - n_xy - |U|, n_rest = |Zs| - |U|.  Strings with n_xy > m are dropped.
    Weight-<=m strings are fixed points.  Preserves <sigma0|.|sigma0> exactly."""
    import itertools
    from scipy.special import comb
    nz = (keys | (keys >> 1)) & _XYMASK
    heavy = np.bitwise_count(nz) > m
    if not heavy.any():
        return keys, coef
    kl, cl = keys[~heavy], coef[~heavy]
    kh, ch = keys[heavy], coef[heavy]
    xy = (kh ^ (kh >> 1)) & _XYMASK
    zz = (kh & (kh >> 1)) & _XYMASK
    nxy = np.bitwise_count(xy).astype(np.int64)
    nzs = np.bitwise_count(zz).astype(np.int64)
    ok = nxy <= m
    kh, ch, xy, zz, nxy, nzs = kh[ok], ch[ok], xy[ok], zz[ok], nxy[ok], nzs[ok]
    kx = kh & (xy | (xy << 1))
    oddmask = np.int64(sum(1 << (2 * i) for i in range(N) if v[i, 3] < 0))
    outk, outc = [kl], [cl]
    for k in range(0, m + 1):
        for sites in itertools.combinations(range(N), k):
            U = np.int64(sum(1 << (2 * i) for i in sites))
            sel = (nxy + k <= m) & ((zz & U) == U)
            if not sel.any():
                continue
            n_rest = nzs[sel] - k
            K = m - nxy[sel] - k
            S = np.where(n_rest == 0, 1.0, ((-1.0) ** K) * comb(np.maximum(n_rest - 1, 0), K))
            sign = 1.0 - 2.0 * (np.bitwise_count((zz[sel] & ~U) & oddmask) & 1)
            outk.append(kx[sel] | U | (U << 1))
            outc.append(ch[sel] * sign * S)
    return np.concatenate(outk), np.concatenate(outc)


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
            if fold is not None:
                nk, nc = fold_strings(nk, nc, fold, N, v)
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
