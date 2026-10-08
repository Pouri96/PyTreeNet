"""Heisenberg-picture operator TEBD on a Pauli-fused MPS (local dimension d = 4, labels I, X, Y, Z).

The operator is O(t) = sum_P c_P P with the Frobenius normalisation sum_P c_P^2 = Tr(O^2)/2^N = 1.  A two-site
gate acts on the coefficients as the real orthogonal 16x16 matrix R of ``pauli_prop.transfer`` (G^dag P G = sum_j R[P, j] P_j).
The circuit is the SAME Strang staircase as the state runs (``mpsenh.run_tebd``); because one step of the staircase is a
palindrome of bonds, the Heisenberg order (reversed circuit) is again 0 -> N-2 -> 0.

Everything here is real float64.  Observables are functions of the Pauli-label marginals
    p_W(alpha) = sum_{P : P|_W = alpha} c_P^2 ,
in particular (single-site W = Z_x or X_x)
    C_Z(x,t) = 2 [p_x(X) + p_x(Y)],   C_X(x,t) = 2 [p_x(Y) + p_x(Z)],   weight density w(x,t) = 1 - p_x(I).

Arms of the cut: 'svd' (renormalised Eckart-Young), 'svdraw' (the same trajectory without renormalisation, see below),
'rw:g' (rTEBD-style label reweighting diag(1, 1/g, 1/g, 1/g) per site, the whole MPS lives in the reweighted frame).
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import _paths  # noqa: F401,E402  (this repository's pytreenet and rule/ onto sys.path)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import mpsenh as M  # noqa: E402
import pauli_prop as PP  # noqa: E402

I2, X, Y, Z = M.I2, M.X, M.Y, M.Z
PL = [I2, X, Y, Z]

# (hx, hz).  'ising' is the repository model (hx = 0.9045, hz = 0.8090); 'tfim' is the hz = 0 null control (free fermions);
# 'ising_hpl' has the two fields swapped, as in Hemery-Pollmann-Luitz.
FIELDS = {'ising': (0.9045, 0.8090), 'tfim': (0.9045, 0.0), 'ising_hpl': (0.8090, 0.9045)}


def h_bond(model, b, N):
    if model in FIELDS:
        hx, hz = FIELDS[model]
        h = np.kron(Z, Z) + np.kron(hx * X + hz * Z, I2)
        if b == N - 2:
            h = h + np.kron(I2, hx * X + hz * Z)
        return h
    if model == 'heis':
        return sum(np.kron(P, P) for P in (X, Y, Z))
    raise ValueError(model)


def make_gates4(model, N, dt):
    """The 4x4 complex bond gates exp(-i dt/2 h_b), identical to ``mpsenh.make_gates`` for the shared models."""
    return [sla.expm(-1j * (dt / 2) * h_bond(model, b, N)) for b in range(N - 1)]


def pauli_gates(model, N, dt, gamma=1.0):
    """Coefficient-space gates G[q_b, q_{b+1}, p_b, p_{b+1}] (new, old) from R.  With gamma > 1 they act in the reweighted
    frame c~ = D c, D = prod_sites diag(1, 1/g, 1/g, 1/g): G~ = D2 R^T D2^{-1}."""
    out = []
    w1 = np.array([1.0, 1 / gamma, 1 / gamma, 1 / gamma])
    for G in make_gates4(model, N, dt):
        R = PP.transfer(G)
        Gop = R.reshape(4, 4, 4, 4).transpose(3, 2, 1, 0).copy()          # R4[p_{b+1}, p_b, q_{b+1}, q_b] -> [q_b, q_{b+1}, p_b, p_{b+1}]
        if gamma != 1.0:
            Gop = Gop * w1[:, None, None, None] * w1[None, :, None, None] / (w1[None, None, :, None] * w1[None, None, None, :])
        out.append(Gop)
    return out


# ------------------------------------------------------------------ MPS
def init_op_mps(N, site=0, pauli=3):
    T = []
    for i in range(N):
        v = np.zeros(4)
        v[pauli if i == site else 0] = 1.0
        T.append(v.reshape(1, 4, 1))
    return T


def mps_to_dense(T):
    v = T[0].reshape(-1, T[0].shape[2])
    for t in T[1:]:
        v = (v @ t.reshape(t.shape[0], -1)).reshape(-1, t.shape[2])
    return v.reshape(-1)


def ranks(T):
    return [t.shape[2] for t in T[:-1]]


def svd_cut_op(theta, chi, dirn):
    """Plain renormalised SVD cut, d = 4 version of ``mpsenh.svd_cut``."""
    l, r = theta.shape[0], theta.shape[3]
    Mm = theta.reshape(l * 4, 4 * r)
    U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
    k = M._rank(s, chi)
    s = s[:k]
    nrm = np.linalg.norm(s)
    U, Vh = U[:, :k], Vh[:k]
    if dirn == 'R':
        return U.reshape(l, 4, k), ((s[:, None] / nrm) * Vh).reshape(k, 4, r), 0.0, False
    return (U * (s / nrm)).reshape(l, 4, k), Vh.reshape(k, 4, r), 0.0, False


class RawSVDCut:
    """SVD cut WITHOUT renormalising the kept centre (HPL-like unnormalised MPO).  Used only to check that 'svdraw' is the
    svd trajectory times a scalar."""
    def __call__(self, theta, chi, dirn, A, B, b):
        l, r = theta.shape[0], theta.shape[3]
        U, s, Vh = np.linalg.svd(theta.reshape(l * 4, 4 * r), full_matrices=False)
        k = M._rank(s, chi)
        s, U, Vh = s[:k], U[:, :k], Vh[:k]
        if dirn == 'R':
            return U.reshape(l, 4, k), (s[:, None] * Vh).reshape(k, 4, r), 0.0, False
        return (U * s).reshape(l, 4, k), Vh.reshape(k, 4, r), 0.0, False


def run_op_tebd(N, chi, nsteps, gops, cut, snap=None, site=0, pauli=3):
    """d-generic copy of ``mpsenh.run_tebd``.  ``cut(theta, chi, dirn, A, B, b) -> (T_b, T_b1, d2, flag)``; ``snap(step, T)`` is
    called after every full step (both sweeps).  Returns the MPS and the wall time."""
    T = init_op_mps(N, site, pauli)
    if hasattr(cut, 'start'):
        cut.start(T)
    t0 = time.time()
    for step in range(nsteps):
        for b in range(N - 1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', gops[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut_op:
                T[b], T[b + 1], _, _ = svd_cut_op(th, chi, 'R')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'R', A, B, b)
        for b in range(N - 2, -1, -1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', gops[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut_op:
                T[b], T[b + 1], _, _ = svd_cut_op(th, chi, 'L')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'L', A, B, b)
        if snap is not None:
            snap(step + 1, T)
    return T, time.time() - t0


# ------------------------------------------------------------------ dense reference (same gates, same order)
def dense_gate(c, G16, b, N):
    """c flat 4^N (C order, site 0 slowest); G16[q, p] = coefficient-space gate with q = 4 q_b + q_{b+1}."""
    A, Zr = 4 ** b, 4 ** (N - b - 2)
    c3 = c.reshape(A, 16, Zr)
    return np.ascontiguousarray(np.tensordot(G16, c3, axes=([1], [1])).transpose(1, 0, 2)).reshape(-1)


def init_dense(N, site=0, pauli=3):
    c = np.zeros(4 ** N)
    c[pauli * 4 ** (N - 1 - site)] = 1.0
    return c


def dense_series(model, N, nsteps, dt, site=0, pauli=3, snap_steps=(), gops=None):
    """Run the exact Heisenberg circuit.  Returns p1 (nsteps+1, N, 4), cz0 (nsteps+1) and the dense c at ``snap_steps``."""
    gops = gops or pauli_gates(model, N, dt)
    G16 = [g.reshape(16, 16) for g in gops]
    c = init_dense(N, site, pauli)
    p1 = np.zeros((nsteps + 1, N, 4))
    cz0 = np.zeros(nsteps + 1)
    idx0 = pauli * 4 ** (N - 1 - site)
    snaps = {}
    p1[0], cz0[0] = marginal1(c, N), c[idx0]
    for step in range(1, nsteps + 1):
        for b in range(N - 1):
            c = dense_gate(c, G16[b], b, N)
        for b in range(N - 2, -1, -1):
            c = dense_gate(c, G16[b], b, N)
        p1[step], cz0[step] = marginal1(c, N), c[idx0]
        if step in snap_steps:
            snaps[step] = c.copy()
    return p1, cz0, snaps


# ------------------------------------------------------------------ scorers (dense side, used for both exact and MPS states)
def marginal1(c, N):
    """p_x(alpha), shape (N, 4), of a flat 4^N coefficient vector (not renormalised)."""
    c2 = (c * c).reshape([4] * N)
    out = np.zeros((N, 4))
    # successive partial sums: sum right-to-left once, left-to-right once
    left = [c2]
    for x in range(N - 1):
        left.append(left[-1].sum(axis=0))
    # left[x] has axes x..N-1; site x marginal = sum over the axes x+1..
    for x in range(N):
        t = left[x]
        out[x] = t.reshape(4, -1).sum(axis=1)
    return out


def marginal_sites(c, N, sites):
    """Joint label marginal over the given sites (sorted), flat 4^len, not renormalised."""
    sites = list(sites)
    c2 = (c * c).reshape([4] * N)
    other = tuple(j for j in range(N) if j not in sites)
    return c2.sum(axis=other).reshape(-1)


def C_from_p1(p1, kind='Z'):
    if kind == 'Z':
        return 2 * (p1[..., 1] + p1[..., 2])
    return 2 * (p1[..., 2] + p1[..., 3])


def weight_from_p1(p1):
    return 1.0 - p1[..., 0]


def phys_dense(T, gamma=1.0):
    """Dense physical coefficient vector of an MPS that lives in the gamma-reweighted frame; returned normalised."""
    c = mps_to_dense(T)
    N = len(T)
    if gamma != 1.0:
        c = c.reshape([4] * N)
        sc = np.array([1.0, gamma, gamma, gamma])
        for i in range(N):
            shp = [1] * N
            shp[i] = 4
            c = c * sc.reshape(shp)
        c = c.reshape(-1)
    return c / np.linalg.norm(c)


def pair_labels(N, dmin=3):
    return [(x, y) for x in range(N) for y in range(x + dmin, N)]
