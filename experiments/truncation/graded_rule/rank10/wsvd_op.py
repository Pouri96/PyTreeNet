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


# ================================================================================================ bond-space factor form (static single cut)
# The same cuts, expressed on the k x k bond matrix diag(s) of an exact two-site tensor theta = U diag(s) Vh.  For k up to 4096 the
# k x k matrices are never formed: the weighted / far-far operators are "diagonal +- low rank" and are applied to blocks of vectors.
# The result is returned as factors  M'_b = Pb Qb^T  (rank <= chi, columns nested in chi) so that the truncated operator is
# U Pb Qb^T Vh and the class errors can be computed without forming the 4096 x 4096 error matrix (see static_cut.py).
def topk_op(matvec, rmatvec, n, K, p=None, maxit=400, tol=1e-13, seed=0, dense=None):
    """Top-K singular triplets of an implicit n x n operator by block subspace iteration with Rayleigh-Ritz.  ``dense`` (optional) is the
    explicit matrix, used when n is small.  Returns Uk (n,K), sk (K,), Vk (n,K), info."""
    if dense is not None and n <= max(3 * K, 400):
        U, s, Vh = np.linalg.svd(dense, full_matrices=False)
        return U[:, :K], s[:K], Vh[:K].T, dict(iters=0, converged=True)
    p = min(n, p or (K + 32))
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(matvec(rng.standard_normal((n, p))))
    sprev = None
    conv = False
    for it in range(1, maxit + 1):
        Z, _ = np.linalg.qr(rmatvec(Q))
        Q, _ = np.linalg.qr(matvec(Z))
        if it % 2 == 0:
            B = rmatvec(Q).T                                   # (p, n) = Q^T W
            Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
            if sprev is not None and np.max(np.abs(s[:K] - sprev[:K])) <= tol * max(s[0], 1e-300):
                conv = True
                break
            sprev = s
    B = rmatvec(Q).T
    Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub[:, :K], s[:K], Vh[:K].T, dict(iters=it, converged=conv)


def _lin(Q1, Qx, a1, a2):
    """X -> X + Q1 diag(a1) Q1^T X + Qx diag(a2) Qx^T X  (symmetric, low-rank perturbation of the identity), a scalars."""
    def f(X):
        out = X.copy()
        if Q1.shape[1]:
            out += a1 * (Q1 @ (Q1.T @ X))
        if Qx.shape[1]:
            out += a2 * (Qx @ (Qx.T @ X))
        return out
    return f


def bond_factors(kind, s, QL1, QLx, QR1, QRx, chi, n=1, lam1=0.0, lam2=0.0, solver=topk_op):
    """Factors (Pb, Qb) (k x chi each) of the truncated bond matrix for the cut ``kind`` in {'svd', 'dmt', 'wsvd'}.
    dmt: Pb = [QL, (I-PL) S QR, Uz sz], Qb = [S QL, QR, Vz], Z = top-(chi - rL - rR) triplets of (I-PL) S (I-PR)   (nested in chi)
    wsvd: top-chi triplets of Lh S Rh, un-whitened with Li, Ri."""
    k = len(s)
    if kind == 'svd':
        r = min(chi, k)
        Pb = np.zeros((k, r))
        Pb[np.arange(r), np.arange(r)] = s[:r]
        Qb = np.zeros((k, r))
        Qb[np.arange(r), np.arange(r)] = 1.0
        return Pb, Qb, dict(iters=0, converged=True)
    if kind == 'wsvd':
        w1, w2 = np.sqrt(1.0 + lam1 + lam2), np.sqrt(1.0 + lam2)
        Lh, Li = _lin(QL1, QLx, w1 - 1, w2 - 1), _lin(QL1, QLx, 1 / w1 - 1, 1 / w2 - 1)
        Rh, Ri = _lin(QR1, QRx, w1 - 1, w2 - 1), _lin(QR1, QRx, 1 / w1 - 1, 1 / w2 - 1)
        # the two perturbations share one weight vector only when QL1 == QR1-structure; they are applied separately on each side
        # (QL1 weight w1 on the near-1 block, w2 on the extra near-2 block), exactly as in OpCut._wsvd
        mv = lambda X: Lh(s[:, None] * Rh(X))
        rmv = lambda X: Rh(s[:, None] * Lh(X))
        dense = None
        if k <= 400:
            dense = Lh(np.diag(s)) @ np.eye(k)
            dense = Rh(dense.T).T
        Uw, sw, Vw, info = solver(mv, rmv, k, chi, dense=dense)
        return Li(Uw * sw), Ri(Vw), info
    if kind == 'dmt':
        QLn = QL1 if n == 1 else np.hstack([QL1, QLx])
        QRn = QR1 if n == 1 else np.hstack([QR1, QRx])
        rL, rR = QLn.shape[1], QRn.shape[1]
        if chi < rL + rR:
            raise InfeasibleCut(f'chi={chi} < rL+rR={rL + rR}')
        fL = lambda X: X - QLn @ (QLn.T @ X)
        fR = lambda X: X - QRn @ (QRn.T @ X)
        mv = lambda X: fL(s[:, None] * fR(X))
        rmv = lambda X: fR(s[:, None] * fL(X))
        rD = chi - rL - rR
        dense = None
        if k <= 400:
            dense = fL(np.diag(s))
            dense = fR(dense.T).T
        if rD > 0:
            Uz, sz, Vz, info = solver(mv, rmv, k, rD, dense=dense)
        else:
            Uz, sz, Vz, info = np.zeros((k, 0)), np.zeros(0), np.zeros((k, 0)), dict(iters=0, converged=True)
        Pb = np.hstack([QLn, fL(s[:, None] * QRn), Uz * sz])
        Qb = np.hstack([s[:, None] * QLn, QRn, Vz])
        return Pb, Qb, info
    raise ValueError(kind)
