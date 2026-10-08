"""d = 4 Pauli-basis operator MPS (Heisenberg picture) used by rank 10: models, gates, MPO helpers, sweep driver.

The operator is O = sum_P c_P P with c_P = Tr[P O] / 2^N real (Hermitian O), basis I, X, Y, Z = 0, 1, 2, 3.  A two-site Trotter gate acts on
the coefficients through the real orthogonal 16x16 matrix R of ``pauli_prop.transfer`` (G^dag P G = sum_q R[P, q] P_q), so the Frobenius
norm of c is conserved by the exact dynamics.  One Strang step is the palindrome [0..N-2, N-2..0]; its reverse is itself, so O(n dt) =
U_step^dag O((n-1) dt) U_step for every n.

Provenance.  The gate convention (``pauli_gates``), the sweep loop with identity-covector environments and the thin-SVD bookkeeping follow
rank4/heis_mpo.py (read-only; its tests/test0.py pass: chi = inf vs dense and vs Pauli propagation to 1e-14).  Everything below is
re-implemented here because rank 10 needs (i) a DMT cut with a general radius n (rank4 has n = 1 only), (ii) the weighted cut, (iii) a model
override ``isingT`` and (iv) rTEBD reweighted frames; the n = 1 DMT is cross-checked against rank4's DMTCut in test_wsvd_op.py.
The only shared state with the repository is ``mpsenh.FIELDS['isingT']`` set in this process (no file is modified).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import _paths  # noqa: F401,E402  (this repository's pytreenet and rule/ onto sys.path)
import mpsenh as M  # noqa: E402
import pauli_prop as PP  # noqa: E402

# OST / DAOE transport model in Pauli units: H = sum Z Z + 1.4 X + 0.9045 Z (2310.06886 g_x = 1.4, g_z = 0.9045).
M.FIELDS.setdefault('isingT', (1.4, 0.9045))

EPS_S = 1e-14          # relative singular-value floor of the thin SVD of theta
LABELS = 'IXYZ'


def fields(model):
    """(g_x, g_z) of the on-site field of an Ising-type model, in Pauli units."""
    return M.FIELDS[model]


# ----------------------------------------------------------------------------------------------- gates
def pauli_gates(model, N, dt, gamma=1.0):
    """Real (4,4,4,4) superoperators S[q_b, q_b1, p_b, p_b1] (new, old) per bond.  With gamma > 1 they act in the rTEBD reweighted frame
    c~ = D c, D = prod_sites diag(1, 1/g, 1/g, 1/g):  S~ = D2 S D2^-1 (no longer orthogonal)."""
    out = []
    w1 = np.array([1.0, 1 / gamma, 1 / gamma, 1 / gamma])
    for G in M.make_gates(model, N, dt):
        R = PP.transfer(G.reshape(4, 4))                        # R[c_in, c_out], c = p_b + 4 p_b1
        S = np.ascontiguousarray(R.reshape(4, 4, 4, 4).transpose(3, 2, 1, 0))
        if gamma != 1.0:
            S = S * w1[:, None, None, None] * w1[None, :, None, None] / (w1[None, None, :, None] * w1[None, None, None, :])
        out.append(S)
    return out


# ----------------------------------------------------------------------------------------------- operators
def eps_matrix(model, x=None):
    """4x4 coefficient matrix [p_x, p_x+1] of the energy density  eps_x = ZZ + g_x/2 (X_x + X_x+1) + g_z/2 (Z_x + Z_x+1)."""
    gx, gz = fields(model)
    m = np.zeros((4, 4))
    m[3, 3] = 1.0
    m[1, 0] = m[0, 1] = gx / 2
    m[3, 0] = m[0, 3] = gz / 2
    return m


def init_local(N, c, mat):
    """MPO of an operator supported on sites (c,) [mat of shape (4,)] or (c, c+1) [mat of shape (4,4)], identity elsewhere."""
    mat = np.asarray(mat, dtype=float)
    T = []
    for i in range(N):
        t = np.zeros((1, 4, 1))
        t[0, 0, 0] = 1.0
        T.append(t)
    if mat.ndim == 1:
        t = np.zeros((1, 4, 1))
        t[0, :, 0] = mat
        T[c] = t
        return T
    U, s, Vh = np.linalg.svd(mat)
    k = max(1, int(np.sum(s > 1e-14 * s[0])))
    T[c] = U[:, :k].reshape(1, 4, k)
    T[c + 1] = (s[:k, None] * Vh[:k]).reshape(k, 4, 1)
    return T


def init_dense_local(N, c, mat):
    """Flat 4^N coefficient vector (C order, site 0 slowest) of the same operator."""
    mat = np.asarray(mat, dtype=float)
    v = np.zeros(4 ** N)
    if mat.ndim == 1:
        for p in range(4):
            v[p * 4 ** (N - 1 - c)] = mat[p]
    else:
        for p in range(4):
            for q in range(4):
                v[p * 4 ** (N - 1 - c) + q * 4 ** (N - 2 - c)] = mat[p, q]
    return v


# ----------------------------------------------------------------------------------------------- MPO helpers
def dense_op(T):
    """Flat 4^N coefficient vector of an MPO."""
    x = T[0][0]
    for t in T[1:]:
        x = np.tensordot(x, t, axes=([-1], [0]))
    return x[..., 0].reshape(-1)


def reweight(T, gamma):
    """Inverse of ``unweight``: physical -> rTEBD frame, c~_P = c_P / gamma^{|P|}."""
    return unweight(T, 1.0 / gamma)


def unweight(T, gamma):
    """Physical-frame copy of a reweighted-frame MPO: c_P = gamma^{|P|} c~_P (identity for gamma == 1)."""
    if gamma == 1.0:
        return T
    out = []
    for t in T:
        u = t.copy()
        u[:, 1:, :] *= gamma
        out.append(u)
    return out


def envs(T):
    """Identity-covector environments: L[i] = contraction of sites < i, R[i] = contraction of sites >= i (L[0] = R[N] = [1])."""
    N = len(T)
    L = [None] * (N + 1)
    R = [None] * (N + 1)
    L[0] = np.ones(1)
    R[N] = np.ones(1)
    for i in range(N):
        L[i + 1] = L[i] @ T[i][:, 0, :]
    for i in range(N - 1, -1, -1):
        R[i] = T[i][:, 0, :] @ R[i + 1]
    return L, R


def local_coeffs(T):
    """c1[s, p] (N,4): coefficient of the single-site string P_s;  c2[s, p, q] (N-1,4,4): coefficient of P_s Q_{s+1}.
    All other sites identity.  Exact contractions through the identity-covector environments."""
    N = len(T)
    L, R = envs(T)
    c1 = np.zeros((N, 4))
    c2 = np.zeros((N - 1, 4, 4))
    for s in range(N):
        c1[s] = np.einsum('l,lpr,r->p', L[s], T[s], R[s + 1])
    for s in range(N - 1):
        c2[s] = np.einsum('l,lpa,aqr,r->pq', L[s], T[s], T[s + 1], R[s + 2])
    return c1, c2


def energy_density(model, c1, c2):
    """C_E(x) = Tr[eps_x O]/2^N for x = 0..N-2 from local coefficients (works for dense and MPO alike)."""
    gx, gz = fields(model)
    N = c1.shape[0]
    x = np.arange(N - 1)
    return c2[x, 3, 3] + gx / 2 * (c1[x, 1] + c1[x + 1, 1]) + gz / 2 * (c1[x, 3] + c1[x + 1, 3])


def local_coeffs_dense(c, N):
    """Same as ``local_coeffs`` from a flat dense coefficient vector."""
    c1 = np.zeros((N, 4))
    c2 = np.zeros((N - 1, 4, 4))
    for s in range(N):
        for p in range(4):
            c1[s, p] = c[p * 4 ** (N - 1 - s)]
    for s in range(N - 1):
        for p in range(4):
            for q in range(4):
                c2[s, p, q] = c[p * 4 ** (N - 1 - s) + q * 4 ** (N - 2 - s)]
    return c1, c2


def dense_gate(c, G16, b, N):
    """Apply the coefficient-space gate G16[q, p] (q = 4 q_b + q_b1) on bond b to a flat 4^N vector."""
    c3 = c.reshape(4 ** b, 16, 4 ** (N - b - 2))
    return np.ascontiguousarray(np.tensordot(G16, c3, axes=([1], [1])).transpose(1, 0, 2)).reshape(-1)


# ----------------------------------------------------------------------------------------------- cuts and sweep
def thin_svd(theta):
    l, r = theta.shape[0], theta.shape[3]
    U, s, Vh = np.linalg.svd(theta.reshape(4 * l, 4 * r), full_matrices=False)
    k = max(1, int(np.sum(s > EPS_S * s[0]))) if s[0] > 0 else 1
    return U[:, :k], s[:k], Vh[:k], s[k:]


def split(U, s, Vh, dirn, l, r):
    k = len(s)
    if dirn == 'R':
        return U.reshape(l, 4, k), (s[:, None] * Vh).reshape(k, 4, r)
    return (U * s).reshape(l, 4, k), Vh.reshape(k, 4, r)


class SVDCut:
    """Plain Frobenius truncation, no renormalisation (the operator norm is not conserved by a truncation and C is read unnormalised)."""
    needs_env = False
    name = 'svd'

    def __call__(self, theta, chi, dirn, ctx):
        l, r = theta.shape[0], theta.shape[3]
        U, s, Vh, tail = thin_svd(theta)
        k = min(chi, len(s))
        disc = float(np.sum(s[k:] ** 2) + np.sum(tail ** 2))
        A, B = split(U[:, :k], s[:k], Vh[:k], dirn, l, r)
        return A, B, disc


def run_heis(model, N, T0, chi, nsteps, dt, cut, gamma=1.0, gates=None, snap=None, hook=None):
    """Strang sweeps of the Heisenberg MPO.  ``cut(theta, chi, dirn, ctx) -> (A, B, disc)``.
    ``snap(n, T)`` is called after every full step with the tensors in the frame the sweep lives in (use ``unweight`` for rw).
    ``hook(b, dirn, n, Tsnap, theta, A, B)`` is called at every cut (tests).  Returns dict(T, disc, wall, maxbond)."""
    S = gates or pauli_gates(model, N, dt, gamma)
    T = [t.copy() for t in (T0 if gamma == 1.0 else reweight(T0, gamma))]       # T0 is given in the physical frame
    needs_env = getattr(cut, 'needs_env', False)
    one = np.ones(1)
    t0 = time.time()
    disc = []
    maxbond = []
    for n in range(1, nsteps + 1):
        stepdisc = 0.0
        if needs_env:
            Lr = [None] * (N + 1)
            Rr = [None] * (N + 1)
            Lr[0] = one
            Rr[N] = one
            for j in range(N - 1, -1, -1):
                Rr[j] = T[j][:, 0, :] @ Rr[j + 1]
        for sweep in ('R', 'L'):
            bonds = range(N - 1) if sweep == 'R' else range(N - 2, -1, -1)
            for b in bonds:
                th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
                th = np.einsum('abst,lstr->labr', S[b], th)
                ctx = dict(b=b, N=N, T=T, Lr=Lr, Rr=Rr, step=n, sweep=sweep) if needs_env else dict(b=b, N=N, step=n, sweep=sweep)
                snp = list(T) if hook is not None else None
                A, B, d = cut(th, chi, sweep, ctx)
                if hook is not None:
                    hook(b, sweep, n, snp, th, A, B)
                T[b], T[b + 1] = A, B
                stepdisc += d
                if needs_env:
                    if sweep == 'R':
                        Lr[b + 1] = Lr[b] @ A[:, 0, :]
                    else:
                        Rr[b + 1] = B[:, 0, :] @ Rr[b + 2]
        disc.append(stepdisc)
        maxbond.append(max(t.shape[2] for t in T[:-1]))
        if snap is not None:
            snap(n, T)
    return dict(T=T, disc=np.array(disc), maxbond=np.array(maxbond), wall=time.time() - t0)
