"""Dense <-> MPS, one compression sweep with a pluggable cut, dense-projection variational fit, and a small numpy two-site DMRG.

Conventions are those of ``mpsenh``: tensors (l, 2, r), site 0 is the most significant bit of the dense vector.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse.linalg as spla
import mpsenh as M


# ---------------------------------------------------------------------------------------------- dense -> MPS
def dense_to_right_mps(v, N, tol=1e-14):
    """exact MPS of the dense vector; sites 1..N-1 right-isometric, site 0 carries the norm"""
    T = [None] * N
    r = 1
    Mx = v.reshape(2 ** (N - 1), 2)
    for i in range(N - 1, 0, -1):
        U, s, Vh = np.linalg.svd(Mx, full_matrices=False)
        k = max(1, int(np.sum(s > tol * s[0])))
        T[i] = Vh[:k].reshape(k, 2, r)
        Mx = (U[:, :k] * s[:k]).reshape(2 ** (i - 1), 2 * k) if i > 1 else (U[:, :k] * s[:k]).reshape(1, 2 * k)
        r = k
    T[0] = Mx.reshape(1, 2, r)
    return T


def sweep_compress(T, chi, cut, direction='R'):
    """one sweep of two-site cuts, no gates: the 'reverse schedule' compression step.  T is modified in place.
    cut is mpsenh.svd_cut or an object called as cut(theta, chi, dirn, A, B, b)."""
    N = len(T)
    if hasattr(cut, 'start'):
        cut.start(T)
    bonds = range(N - 1) if direction == 'R' else range(N - 2, -1, -1)
    for b in bonds:
        th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
        A = T[b - 1] if b >= 1 else None
        B = T[b + 2] if b + 2 <= N - 1 else None
        if cut is M.svd_cut:
            T[b], T[b + 1], _, _ = M.svd_cut(th, chi, direction)
        else:
            T[b], T[b + 1], _, _ = cut(th, chi, direction, A, B, b)
    return T


def ranks(T):
    return [t.shape[2] for t in T[:-1]]


def to_dense(T):
    v = M.mps_to_dense(T)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------------------------- variational fit (dense projection)
def _right_canon(T):
    N = len(T)
    T = [t.copy() for t in T]
    for i in range(N - 1, 0, -1):
        l, _, r = T[i].shape
        Q, R = np.linalg.qr(T[i].reshape(l, 2 * r).T)       # T = R^T Q^T, Q^T right isometric
        k = Q.shape[1]
        T[i] = Q.T.reshape(k, 2, r)
        T[i - 1] = np.tensordot(T[i - 1], R.T, axes=([2], [0]))
    T[0] = T[0] / np.linalg.norm(T[0])
    return T


def _right_envs(T):
    N = len(T)
    Rb = [None] * (N + 1)
    Rb[N] = np.ones((1, 1), dtype=complex)
    for i in range(N - 1, -1, -1):
        Rb[i] = np.tensordot(T[i], Rb[i + 1], axes=([2], [0])).reshape(T[i].shape[0], -1)
    return Rb


def _left_envs(T):
    N = len(T)
    Lb = [None] * (N + 1)
    Lb[0] = np.ones((1, 1), dtype=complex)
    for i in range(N):
        Lb[i + 1] = np.einsum('al,lsr->asr', Lb[i], T[i]).reshape(-1, T[i].shape[2])
    return Lb


def var_fit(psi, T0, nsweeps=3):
    """maximise |<psi|phi>| over MPS phi with (at most) the bond dimensions of T0 by one-site ALS sweeps (the exact one-site optimum at
    every step).  psi: normalised dense vector.  Returns the fitted MPS and the fidelity after each half sweep."""
    N = len(T0)
    T = _right_canon(T0)
    psi = psi / np.linalg.norm(psi)
    hist = []

    def fid():
        v = M.mps_to_dense(T)
        return float(abs(np.vdot(psi, v / np.linalg.norm(v))) ** 2)

    def centre(Lbi, Rbi1, i):
        X = Lbi.conj().T @ psi.reshape(2 ** i, -1)                       # (l, 2 * 2^(N-i-1))
        X = X.reshape(Lbi.shape[1], 2, -1)
        return np.einsum('lsb,rb->lsr', X, Rbi1.conj())

    for sw in range(nsweeps):
        Rb = _right_envs(T)
        Lb = np.ones((1, 1), dtype=complex)
        for i in range(N):
            C = centre(Lb, Rb[i + 1], i)
            l, _, r = C.shape
            if i < N - 1:
                Q, _R = np.linalg.qr(C.reshape(l * 2, r))
                T[i] = Q.reshape(l, 2, Q.shape[1])
                Lb = np.einsum('al,lsr->asr', Lb, T[i]).reshape(-1, T[i].shape[2])
            else:
                T[i] = C
        hist.append(fid())
        Lbs = _left_envs(T)
        Rb = np.ones((1, 1), dtype=complex)
        for i in range(N - 1, -1, -1):
            C = centre(Lbs[i], Rb, i)
            l, _, r = C.shape
            if i > 0:
                Q, _R = np.linalg.qr(C.reshape(l, 2 * r).T)
                T[i] = Q.T.reshape(Q.shape[1], 2, r)
                Rb = np.tensordot(T[i], Rb, axes=([2], [0])).reshape(T[i].shape[0], -1)
            else:
                T[i] = C
        hist.append(fid())
    v = M.mps_to_dense(T)
    return T, hist


# ---------------------------------------------------------------------------------------------- MPO and two-site DMRG
def nn_mpo(N, terms):
    """MPO (list of (Dl, Dr, 2, 2) arrays, physical indices (out, in)) of a sum of one-site and nearest-neighbour two-site Pauli strings"""
    P = {'X': M.X, 'Y': M.Y, 'Z': M.Z}
    pairs = []
    for c, t in terms:
        if len(t) == 2:
            assert t[1][0] == t[0][0] + 1, 'nearest neighbour terms only'
            key = (t[0][1], t[1][1])
            if key not in pairs:
                pairs.append(key)
    K = len(pairs)
    D = K + 2
    W = [np.zeros((D, D, 2, 2), dtype=complex) for _ in range(N)]
    for i in range(N):
        W[i][0, 0] = M.I2
        W[i][D - 1, D - 1] = M.I2
        for k, (a, b) in enumerate(pairs):
            W[i][0, 1 + k] = P[a]
    field = [np.zeros((2, 2), dtype=complex) for _ in range(N)]
    for c, t in terms:
        if len(t) == 1:
            field[t[0][0]] += c * P[t[0][1]]
        elif len(t) == 2:
            k = pairs.index((t[0][1], t[1][1]))
            W[t[1][0]][1 + k, D - 1] += c * P[t[1][1]]
    for i in range(N):
        W[i][0, D - 1] += field[i]
    W[0] = W[0][0:1]
    W[N - 1] = W[N - 1][:, D - 1:D]
    return W


def _mpo_expect(T, W):
    N = len(T)
    E = np.ones((1, 1, 1), dtype=complex)
    for i in range(N):
        E = np.einsum('abc,asd,bBst,ctf->dBf', E, T[i].conj(), W[i], T[i])
    return float(E.reshape(-1)[0].real)


def dmrg2(W, N, chi, nsweeps=12, seed=0, tol=1e-11, T_init=None, verbose=False):
    """two-site DMRG with bond dimension cap chi; returns the MPS (centre at site 0) and the true energy after every sweep"""
    rng = np.random.default_rng(seed)
    if T_init is None:
        dims = [min(chi, 2 ** min(i, N - i)) for i in range(N + 1)]
        T = [(rng.normal(size=(dims[i], 2, dims[i + 1])) + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))) for i in range(N)]
    else:
        T = [t.copy() for t in T_init]
    T = _right_canon(T)
    Lenv = [None] * (N + 1)
    Renv = [None] * (N + 1)
    Lenv[0] = np.ones((1, 1, 1), dtype=complex)
    Renv[N] = np.ones((1, 1, 1), dtype=complex)

    def upd_R(i):
        Renv[i] = np.einsum('asd,bBst,ctf,dBf->abc', T[i].conj(), W[i], T[i], Renv[i + 1])

    def upd_L(i):
        Lenv[i + 1] = np.einsum('abc,asd,bBst,ctf->dBf', Lenv[i], T[i].conj(), W[i], T[i])

    for i in range(N - 1, 0, -1):
        upd_R(i)
    hist = []
    for sw in range(nsweeps):
        for dirn in ('R', 'L'):
            bonds = range(N - 1) if dirn == 'R' else range(N - 2, -1, -1)
            for b in bonds:
                th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))       # (l,2,2,r)
                l, r = th.shape[0], th.shape[3]

                def mv(x, Lb=Lenv[b], Rb=Renv[b + 2], W1=W[b], W2=W[b + 1], l=l, r=r):
                    x = x.reshape(l, 2, 2, r)
                    y = np.einsum('LBl,lstr->LBstr', Lb, x)
                    y = np.einsum('LBstr,BCus->LCutr', y, W1)
                    y = np.einsum('LCutr,CDvt->LDuvr', y, W2)
                    y = np.einsum('LDuvr,RDr->LuvR', y, Rb)
                    return y.reshape(-1)
                op = spla.LinearOperator((l * 4 * r, l * 4 * r), matvec=mv, dtype=complex)
                w, V = spla.eigsh(op, k=1, which='SA', v0=th.reshape(-1), tol=1e-12)
                th = V[:, 0].reshape(l * 2, 2 * r)
                U, s, Vh = np.linalg.svd(th, full_matrices=False)
                k = min(chi, int(np.sum(s > 1e-14 * s[0])))
                s = s[:k] / np.linalg.norm(s[:k])
                if dirn == 'R':
                    T[b] = U[:, :k].reshape(l, 2, k)
                    T[b + 1] = (s[:, None] * Vh[:k]).reshape(k, 2, r)
                    upd_L(b)
                else:
                    T[b] = (U[:, :k] * s).reshape(l, 2, k)
                    T[b + 1] = Vh[:k].reshape(k, 2, r)
                    upd_R(b + 1)
        e = _mpo_expect(T, W) / float(np.sum(np.abs(M.mps_to_dense(T)) ** 2)) if N <= 20 else _mpo_expect(T, W)
        hist.append(e)
        if verbose:
            print(sw, e, ranks(T))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
    return T, hist
