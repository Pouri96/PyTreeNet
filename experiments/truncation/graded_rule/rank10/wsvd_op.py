"""Cut-local truncations of a Pauli-basis operator MPO / bond matrix:  DMT-n (reserved rank, identity covector) and the weighted cut
``wsvd`` (closed-form window-whitened SVD).

Geometry.  Cut between sites b and b+1 of the operator O = sum_P c_P P.  The bond matrix M[i, j] = c_{(i, j)} has rows i = left half-string
(p_0..p_b) and columns j = right half-string (p_{b+1}..p_{N-1}).  With the IDENTITY covector on the far sites the "near-n" left rows are the
strings that are identity on sites < b-n+1 (4^n of them), the near-n right columns those that are identity on sites > b+n (4^n).  A string
straddling the cut has left extent a (distance from the cut to the leftmost non-identity site, a >= 1 means non-identity at b-a+1) and right
extent c; span = a + c.
  * DMT-n (1707.01506 / 1902.01859, traceless Heisenberg form of 2310.06886 App. D 2, no M_00 division) keeps every row of M in the
    near-n left set (all columns) and every column in the near-n right set (all rows) exactly, and SVD-truncates only the far-far block to
    rank chi - rL - rR.  It therefore preserves exactly every functional supported on sites [b-n+1, N-1] or [0, b+n], i.e. strings
    with a <= n or c <= n.  Reserved rank 2*4^n (8 for n = 1, 32 for n = 2).
  * wsvd:  min_{rank M' <= chi} || L^{1/2} (M - M') R^{1/2} ||_F  with the separable metric L = I + lam1 P1_L + lam2 P2_L (P_n = projector on
    the near-n rows), R likewise.  Closed form: SVD of L^{1/2} M R^{1/2} (Eckart-Young), then un-whiten.  lam = 0 is plain SVD, lam -> inf
    forces the near rows/columns to be reproduced exactly (the DMT constraint) without reserving rank.
  * rw:g (rTEBD, 2412.08730) is the same closed form with the global weights D = g^{-weight} (see ``weights_rw``).
Everything is orthogonal-basis based, so the Frobenius norm of c is the Hilbert-Schmidt norm of O.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pmpo as P  # noqa: E402


class InfeasibleCut(Exception):
    """DMT-n needs chi >= rL + rR."""


# ================================================================================================ MPO-level cuts
def orth_cols(R, rtol=1e-13):
    """Orthonormal basis (k x r) of the column space of R (k x m)."""
    if R.shape[1] == 0:
        return np.zeros((R.shape[0], 0))
    U, sv, _ = np.linalg.svd(R, full_matrices=False)
    if sv[0] <= 1e-300:
        return U[:, :0]
    return U[:, :int(np.sum(sv > rtol * sv[0]))]


def nested_basis(R1, R2, rtol=1e-13):
    """Q1 spans col(R1); Qx completes it to col(R2) (R1's columns are a subset of R2's).  Qx is empty if R2 is None."""
    Q1 = orth_cols(R1, rtol)
    if R2 is None:
        return Q1, np.zeros((Q1.shape[0], 0))
    res = R2 - Q1 @ (Q1.T @ R2)
    nrm = np.linalg.norm(R2)
    if nrm <= 1e-300:
        return Q1, np.zeros((Q1.shape[0], 0))
    U, sv, _ = np.linalg.svd(res, full_matrices=False)
    r = int(np.sum(sv > rtol * nrm)) if sv.size else 0
    return Q1, U[:, :r]


def complete(Q):
    """Orthogonal k x k matrix whose first r columns span col(Q)."""
    k, r = Q.shape
    if r == 0:
        return np.eye(k)
    Qf, _ = np.linalg.qr(Q, mode='complete')
    return Qf


def near_spaces(ctx, U, Vh, l, r, nmax=2):
    """Bond-space functionals of the near-1 / near-2 operators on both sides of the cut (identity covector on the far sites).
    Returns RL1 (k,4), RL2 (k,16) or None, RR1 (k,4), RR2 (k,16) or None.  Column space(RL1) is contained in column space(RL2)."""
    b, N, T, Lr, Rr = ctx['b'], ctx['N'], ctx['T'], ctx['Lr'], ctx['Rr']
    k = U.shape[1]
    Ut = U.reshape(l, 4, k)
    Vt = Vh.reshape(k, 4, r)
    RL1 = np.einsum('l,lqk->kq', Lr[b], Ut)
    RR1 = np.einsum('kpr,r->kp', Vt, Rr[b + 2])
    RL2 = RR2 = None
    if nmax >= 2 and b >= 1:
        Wl = np.einsum('x,xpl->lp', Lr[b - 1], T[b - 1])                    # (l, p_{b-1})
        RL2 = np.einsum('lqk,lp->kpq', Ut, Wl).reshape(k, 16)
    if nmax >= 2 and b + 2 <= N - 1:
        Wr = np.einsum('rqs,s->rq', T[b + 2], Rr[b + 3])                    # (r, q_{b+2})
        RR2 = np.einsum('kpr,rq->kpq', Vt, Wr).reshape(k, 16)
    return RL1, RL2, RR1, RR2


class OpCut:
    """Cut of the two-site tensor theta.  kind = 'dmt' (radius n) or 'wsvd' (lam1 on the near-1 block, lam2 on the near-2 block)."""
    needs_env = True

    def __init__(self, kind, n=1, lam1=0.0, lam2=0.0, tol=1e-13):
        assert kind in ('dmt', 'wsvd')
        self.kind, self.n, self.lam1, self.lam2, self.tol = kind, n, float(lam1), float(lam2), tol
        self.name = f'dmt:{n}' if kind == 'dmt' else f'wsvd:{lam1:g}:{lam2:g}'
        self.ntrunc = 0
        self.reserved = []
        if kind == 'wsvd' and lam1 == 0 and lam2 == 0:
            self.needs_env = False

    def __call__(self, theta, chi, dirn, ctx):
        l, r = theta.shape[0], theta.shape[3]
        if self.kind == 'wsvd' and self.lam1 == 0 and self.lam2 == 0:
            return P.SVDCut()(theta, chi, dirn, ctx)                       # lam = 0 is the plain SVD cut, bit for bit
        U, s, Vh, tail = P.thin_svd(theta)
        k = len(s)
        if k <= chi:
            A, B = P.split(U, s, Vh, dirn, l, r)
            return A, B, float(np.sum(tail ** 2))
        self.ntrunc += 1
        RL1, RL2, RR1, RR2 = near_spaces(ctx, U, Vh, l, r, 2)
        QL1, QLx = nested_basis(RL1, RL2, self.tol)
        QR1, QRx = nested_basis(RR1, RR2, self.tol)
        if self.kind == 'dmt':
            return self._dmt(U, s, Vh, tail, chi, dirn, l, r, QL1, QLx, QR1, QRx)
        return self._wsvd(U, s, Vh, tail, chi, dirn, l, r, QL1, QLx, QR1, QRx)

    def _dmt(self, U, s, Vh, tail, chi, dirn, l, r, QL1, QLx, QR1, QRx):
        QLn = QL1 if self.n == 1 else np.hstack([QL1, QLx])
        QRn = QR1 if self.n == 1 else np.hstack([QR1, QRx])
        rL, rR = QLn.shape[1], QRn.shape[1]
        if chi < rL + rR:
            raise InfeasibleCut(f'chi={chi} < rL+rR={rL + rR}')
        QL, QR = complete(QLn), complete(QRn)
        rD = chi - rL - rR
        Mt = (QL.T * s) @ QR
        Ud, sd, Vd = np.linalg.svd(Mt[rL:, rR:], full_matrices=False)
        Mt2 = Mt.copy()
        Mt2[rL:, rR:] = (Ud[:, :rD] * sd[:rD]) @ Vd[:rD]
        disc = float(np.sum(sd[rD:] ** 2) + np.sum(tail ** 2))
        self.reserved.append((rL, rR))
        return self._finish(U, Vh, Mt2, QL, QR, chi, dirn, l, r, disc)

    def _wsvd(self, U, s, Vh, tail, chi, dirn, l, r, QL1, QLx, QR1, QRx):
        w1, w2 = np.sqrt(1.0 + self.lam1 + self.lam2), np.sqrt(1.0 + self.lam2)
        k = len(s)

        def halves(Q1, Qx):
            Lh = np.eye(k) + (w1 - 1) * (Q1 @ Q1.T) + (w2 - 1) * (Qx @ Qx.T)
            Li = np.eye(k) + (1 / w1 - 1) * (Q1 @ Q1.T) + (1 / w2 - 1) * (Qx @ Qx.T)
            return Lh, Li
        Lh, Li = halves(QL1, QLx)
        Rh, Ri = halves(QR1, QRx)
        W = Lh @ (s[:, None] * Rh)
        Uw, sw, Vw = np.linalg.svd(W, full_matrices=False)
        Mp = Li @ ((Uw[:, :chi] * sw[:chi]) @ Vw[:chi]) @ Ri                 # M' in the (U, Vh) basis, rank <= chi
        disc = float(np.sum((np.diag(s) - Mp) ** 2) + np.sum(tail ** 2))
        return self._finish(U, Vh, Mp, np.eye(k), np.eye(k), chi, dirn, l, r, disc)

    @staticmethod
    def _finish(U, Vh, Mt2, QL, QR, chi, dirn, l, r, disc):
        U2, s2, V2 = np.linalg.svd(Mt2, full_matrices=False)
        k2 = max(1, min(chi, int(np.sum(s2 > P.EPS_S * s2[0])))) if s2[0] > 0 else 1
        Un = U @ (QL @ U2[:, :k2])
        Vn = (V2[:k2] @ QR.T) @ Vh
        A, B = P.split(Un, s2[:k2], Vn, dirn, l, r)
        return A, B, disc


def make_cut(spec):
    """'svd' | 'dmt:1' | 'dmt:2' | 'wsvd:LAM1:LAM2' | 'rw:G' (rw is a reweighted-frame SVD, handled by the driver: returns the SVD cut)."""
    p = spec.split(':')
    if p[0] in ('svd', 'rw'):
        return P.SVDCut()
    if p[0] == 'dmt':
        return OpCut('dmt', n=int(p[1]))
    if p[0] == 'wsvd':
        return OpCut('wsvd', lam1=float(p[1]), lam2=float(p[2]))
    raise ValueError(spec)


def spec_gamma(spec):
    p = spec.split(':')
    return float(p[1]) if p[0] == 'rw' else 1.0


# ================================================================================================ static (dense bond matrix) tools
def topk_svd(A, K, p=None, maxit=80, tol=1e-12, seed=0):
    """Top-K singular triplets of a dense A by block subspace iteration with Rayleigh-Ritz (block size p >= K).  Falls back to the dense
    SVD for small matrices.  Returns U (m,K), s (K,), Vh (K,n), niter."""
    m, n = A.shape
    if min(m, n) <= max(2 * K, 256):
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        return U[:, :K], s[:K], Vh[:K], 0
    p = p or (K + 24)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(A @ rng.standard_normal((n, p)))
    sprev = None
    for it in range(1, maxit + 1):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
        if it % 2 == 0 or it == maxit:
            Bm = Q.T @ A
            Ub, s, Vh = np.linalg.svd(Bm, full_matrices=False)
            if sprev is not None and np.max(np.abs(s[:K] - sprev[:K])) <= tol * s[0]:
                break
            sprev = s
    Bm = Q.T @ A
    Ub, s, Vh = np.linalg.svd(Bm, full_matrices=False)
    return Q @ Ub[:, :K], s[:K], Vh[:K], it


def static_layout(N, b):
    """Row/column class bookkeeping for the cut between sites b and b+1 of an N-site operator (C order, site 0 slowest).
    Returns dict: nrow, ncol, a (left extent per row), c (right extent per column), weight_row, weight_col (Pauli weights)."""
    nL, nR = b + 1, N - b - 1
    nrow, ncol = 4 ** nL, 4 ** nR
    i = np.arange(nrow)
    dl = np.stack([(i // 4 ** (nL - 1 - s)) % 4 for s in range(nL)], axis=1)          # digit of site s
    nz = dl != 0
    a = np.where(nz.any(1), nL - np.argmax(nz, axis=1), 0)                              # leftmost non-identity: a = b - s_min + 1
    j = np.arange(ncol)
    dr = np.stack([(j // 4 ** (nR - 1 - s)) % 4 for s in range(nR)], axis=1)           # digit of site b+1+s
    nzr = dr != 0
    c = np.where(nzr.any(1), nR - np.argmax(nzr[:, ::-1], axis=1), 0)                   # rightmost non-identity: c = s_max + 1
    return dict(nrow=nrow, ncol=ncol, a=a, c=c, wrow=nz.sum(1), wcol=nzr.sum(1), nL=nL, nR=nR)


def near_idx(lay, n):
    """Indices of the near-n rows (first 4^n) and near-n columns (4^n strings with the n sites next to the cut free)."""
    rows = np.arange(4 ** n)
    cols = np.arange(4 ** n) * 4 ** (lay['nR'] - n)
    return rows, cols


def weights_wsvd(lay, lam1, lam2):
    wl = np.ones(lay['nrow'])
    wr = np.ones(lay['ncol'])
    r1, c1 = near_idx(lay, 1)
    r2, c2 = near_idx(lay, 2)
    w1, w2 = np.sqrt(1 + lam1 + lam2), np.sqrt(1 + lam2)
    wl[r2] = w2
    wr[c2] = w2
    wl[r1] = w1
    wr[c1] = w1
    return wl, wr


def weights_rw(lay, gamma):
    return gamma ** (-lay['wrow'].astype(float)), gamma ** (-lay['wcol'].astype(float))


class StaticFactors:
    """Low-rank factors of the truncated bond matrices for chi in ``chis``:  M'(chi) = Pl[:, :chi] @ Qr[:chi]  (weighted arms)."""

    def __init__(self, Mx, wl, wr, K, solver=topk_svd):
        W = (wl[:, None] * Mx) * wr[None, :]
        U, s, Vh, self.niter = solver(W, K)
        self.s = s
        self.Pl = (U * s) / wl[:, None]
        self.Qr = Vh / wr[None, :]
        self.wnorm2 = float(np.sum(W * W))

    def error(self, Mx, chi):
        return Mx - self.Pl[:, :chi] @ self.Qr[:chi]


class StaticDMT:
    """DMT-n on a dense bond matrix: near rows/columns kept exactly, the far-far block SVD-truncated to chi - 2*4^n."""

    def __init__(self, Mx, lay, n, K, solver=topk_svd):
        self.n = n
        self.rows, self.cols = near_idx(lay, n)
        self.far_r = np.setdiff1d(np.arange(lay['nrow']), self.rows)
        self.far_c = np.setdiff1d(np.arange(lay['ncol']), self.cols)
        self.D = Mx[np.ix_(self.far_r, self.far_c)]
        self.reserved = len(self.rows) + len(self.cols)
        self.Kd = max(1, K - self.reserved)
        self.U, self.s, self.Vh, self.niter = solver(self.D, self.Kd)

    def feasible(self, chi):
        return chi >= self.reserved

    def error(self, Mx, chi):
        rD = chi - self.reserved
        Dr = (self.U[:, :rD] * self.s[:rD]) @ self.Vh[:rD] if rD > 0 else 0.0
        E = np.zeros_like(Mx)
        E[np.ix_(self.far_r, self.far_c)] = self.D - Dr
        return E


class ClassScorer:
    """Squared coefficient error summed over the (a, c) classes of strings, C(x) overlaps, Frobenius error."""

    def __init__(self, N, b, model, Mx, c0_site):
        self.N, self.b, self.model = N, b, model
        self.lay = static_layout(N, b)
        lay = self.lay
        self.amax, self.cmax = lay['nL'], lay['nR']
        self.Ar = (lay['a'][None, :] == np.arange(self.amax + 1)[:, None]).astype(float)
        self.Ac = (lay['c'][None, :] == np.arange(self.cmax + 1)[:, None]).astype(float)
        self.M2 = self.class_sq(Mx)
        self.fro2 = float(np.sum(Mx * Mx))
        # flat positions of the strings entering eps_x (x = 0..N-2) and Z_x
        self.ncol = lay['ncol']
        self.Mx_C = self.C_of(Mx)

    def class_sq(self, E):
        return self.Ar @ ((E * E) @ self.Ac.T)

    def _pos(self, sites_paulis):
        flat = sum(p * 4 ** (self.N - 1 - s) for s, p in sites_paulis)
        return divmod(flat, self.ncol)

    def C_of(self, Mx):
        N = self.N
        gx, gz = P.fields(self.model)
        out = np.zeros(N - 1)
        for x in range(N - 1):
            def g(sp):
                i, j = self._pos(sp)
                return Mx[i, j]
            out[x] = g([(x, 3), (x + 1, 3)]) + gx / 2 * (g([(x, 1)]) + g([(x + 1, 1)])) + gz / 2 * (g([(x, 3)]) + g([(x + 1, 3)]))
        return out

    def score(self, E):
        S = self.class_sq(E)
        return dict(S=S.tolist(), fro2=float(np.sum(E * E)), dC=(self.C_of(E)).tolist())


def span_classes(S, S_ref=None):
    """Collapse an (a, c) table of squared errors into the plan's span classes over straddling strings (a >= 1, c >= 1):
    le3, s4, s5, ge6, plus the DMT-protection classes unprot1 (a>=2 and c>=2), unprot2 (a>=3 and c>=3) and the one-sided strings."""
    S = np.asarray(S)
    A, C = S.shape
    out = dict(le3=0.0, s4=0.0, s5=0.0, ge6=0.0, unprot1=0.0, unprot2=0.0, onesided=0.0)
    for a in range(A):
        for c in range(C):
            v = S[a, c]
            if a == 0 or c == 0:
                out['onesided'] += v
                continue
            sp = a + c
            out['le3' if sp <= 3 else 's4' if sp == 4 else 's5' if sp == 5 else 'ge6'] += v
            if a >= 2 and c >= 2:
                out['unprot1'] += v
            if a >= 3 and c >= 3:
                out['unprot2'] += v
    return out
