"""Static single-cut mode (Test 1): truncate ONE cut of an exact Heisenberg operator and score the coefficient errors by string class.

The exact operator is a dense Pauli vector c (4^N, from heis_ref); cutting between sites b and b+1 acts on the bond matrix M (4^(b+1) x
4^(N-b-1)).  The cut is applied in the bond gauge of the exact canonical MPO, i.e. in the basis of the singular vectors of M (U, s, V): the
near-n sets are the bond-space functionals U[near rows, :]^T, V[near cols, :]^T, exactly what OpCut computes for a mixed-canonical exact MPO
(test_wsvd_op.py checks this).  Error tables are computed from low-rank factors without forming the error matrix.

Class convention: a string straddling the cut has left extent a >= 1 and right extent c >= 1 (distance of its outermost non-identity sites
from the cut), span = a + c.  DMT-n preserves exactly every string with a <= n or c <= n (incl. one-sided strings a = 0 or c = 0).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pmpo as P  # noqa: E402
import wsvd_op as W  # noqa: E402


def static_layout(N, b):
    """Row/column class bookkeeping (C order, site 0 slowest): a (left extent), c (right extent), Pauli weights, sizes."""
    nL, nR = b + 1, N - b - 1
    nrow, ncol = 4 ** nL, 4 ** nR
    i = np.arange(nrow)
    dl = np.stack([(i // 4 ** (nL - 1 - s)) % 4 for s in range(nL)], axis=1)          # digit of site s
    nz = dl != 0
    a = np.where(nz.any(1), nL - np.argmax(nz, axis=1), 0)                              # leftmost non-identity site s_min: a = b - s_min + 1
    j = np.arange(ncol)
    dr = np.stack([(j // 4 ** (nR - 1 - s)) % 4 for s in range(nR)], axis=1)           # digit of site b+1+s
    nzr = dr != 0
    c = np.where(nzr.any(1), nR - np.argmax(nzr[:, ::-1], axis=1), 0)                   # rightmost non-identity site b+1+s_max: c = s_max + 1
    return dict(nrow=nrow, ncol=ncol, a=a, c=c, wrow=nz.sum(1), wcol=nzr.sum(1), nL=nL, nR=nR)


def near_idx(lay, n):
    """Near-n rows (first 4^n) and near-n columns (the n sites next to the cut free, identity beyond)."""
    return np.arange(4 ** n), np.arange(4 ** n) * 4 ** (lay['nR'] - n)


def span_classes(S):
    """Collapse an (a, c) table of squared errors into the plan's span classes over straddling strings, plus the DMT-protection classes."""
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


class StaticCut:
    def __init__(self, c, N, b, model, verbose=False, kcut=1e-12):
        self.N, self.b, self.model = N, b, model
        self.lay = lay = static_layout(N, b)
        self.M = c.reshape(lay['nrow'], lay['ncol'])
        t0 = time.time()
        U, s, Vh = np.linalg.svd(self.M, full_matrices=False)
        k = max(1, int(np.sum(s > kcut * s[0])))          # bond space of the 'exact' MPO: singular values above kcut (class errors are insensitive to kcut in 1e-8..1e-14)
        self.U, self.s, self.V = U[:, :k], s[:k], Vh[:k].T.copy()
        del U, Vh
        self.k = k
        self.t_svd = time.time() - t0
        if verbose:
            print(f'    bond SVD {lay["nrow"]}x{lay["ncol"]}: k={k}  {self.t_svd:.0f}s', flush=True)
        A, C = lay['nL'] + 1, lay['nR'] + 1
        self.rows_by_a = [np.where(lay['a'] == a)[0] for a in range(A)]
        self.cols_by_c = [np.where(lay['c'] == cc)[0] for cc in range(C)]
        self.M2 = np.array([[np.sum(self.M[np.ix_(ra, rc)] ** 2) for rc in self.cols_by_c] for ra in self.rows_by_a])
        self.fro2 = float(np.sum(self.M ** 2))
        rows1, cols1 = near_idx(lay, 1)
        rows2, cols2 = near_idx(lay, 2)
        self.QL1, self.QLx = W.nested_basis(self.U[rows1].T, self.U[rows2].T)
        self.QR1, self.QRx = W.nested_basis(self.V[cols1].T, self.V[cols2].T)
        self.rL, self.rR = (self.QL1.shape[1], self.QLx.shape[1]), (self.QR1.shape[1], self.QRx.shape[1])
        self.pos = self._positions()

    def _positions(self):
        """(row, col) of the strings entering eps_x and Z_x."""
        N, ncol = self.N, self.lay['ncol']

        def pos(sp):
            return divmod(sum(p * 4 ** (N - 1 - s) for s, p in sp), ncol)
        out = []
        for x in range(N - 1):
            out.append([(pos([(x, 3), (x + 1, 3)]), 1.0), (pos([(x, 1)]), None), (pos([(x + 1, 1)]), None), (pos([(x, 3)]), None), (pos([(x + 1, 3)]), None)])
        return out

    def C_of_factors(self, P_, Q_):
        """Error in C(x) = Tr[eps_x O]/2^N for the truncated operator minus exact, x = 0..N-2, from factors (rows of P_ and Q_)."""
        gx, gz = P.fields(self.model)
        wts = [1.0, gx / 2, gx / 2, gz / 2, gz / 2]
        out = np.zeros(self.N - 1)
        for x, lst in enumerate(self.pos):
            for w, ((i, j), _) in zip(wts, lst):
                out[x] += w * (self.M[i, j] - P_[i] @ Q_[j])
        return out

    # ------------------------------------------------------------------------------------------ arms
    def arm_factors(self, spec, chi_max):
        """Pauli-coordinate factors (Pm, Qm) with M'(chi) = Pm[:, :chi] Qm[:, :chi]^T (chi >= reserved rank for dmt), plus solver info."""
        p = spec.split(':')
        if p[0] == 'rw':
            g = float(p[1])
            wl, wr = g ** (-self.lay['wrow'].astype(float)), g ** (-self.lay['wcol'].astype(float))
            Mw = (wl[:, None] * self.M) * wr[None, :]
            Uw, sw, Vw, info = self._rect_topk(Mw, chi_max)
            return (Uw * sw) / wl[:, None], Vw / wr[:, None], info
        if p[0] == 'svd':
            kind, kw = 'svd', {}
        elif p[0] == 'dmt':
            kind, kw = 'dmt', dict(n=int(p[1]))
        elif p[0] == 'wsvd':
            kind, kw = 'wsvd', dict(lam1=float(p[1]), lam2=float(p[2]))
        else:
            raise ValueError(spec)
        Pb, Qb, info = W.bond_factors(kind, self.s, self.QL1, self.QLx, self.QR1, self.QRx, chi_max, **kw)
        return self.U @ Pb, self.V @ Qb, info

    @staticmethod
    def _rect_topk(Mw, K):
        """Top-K of a rectangular matrix via the small-side Gram trick is avoided; use block iteration with explicit matvecs on the long side."""
        m, n = Mw.shape
        rng = np.random.default_rng(0)
        p = K + 32
        Q, _ = np.linalg.qr(Mw @ rng.standard_normal((n, p)))
        sprev, conv = None, False
        for it in range(1, 401):
            Z, _ = np.linalg.qr(Mw.T @ Q)
            Q, _ = np.linalg.qr(Mw @ Z)
            if it % 2 == 0:
                Ub, s, Vh = np.linalg.svd((Mw.T @ Q).T, full_matrices=False)
                if sprev is not None and np.max(np.abs(s[:K] - sprev[:K])) <= 1e-13 * s[0]:
                    conv = True
                    break
                sprev = s
        Ub, s, Vh = np.linalg.svd((Mw.T @ Q).T, full_matrices=False)
        return Q @ Ub[:, :K], s[:K], Vh[:K].T, dict(iters=it, converged=conv)

    # ------------------------------------------------------------------------------------------ scoring
    def tables(self, Pm, Qm, chis):
        """Squared-error class tables S[a, c] and C(x) errors for each chi (prefix columns), via
        S = M2 - 2 <M_ac, P_a Q_c^T> + ||P_a Q_c^T||^2."""
        lay, M = self.lay, self.M
        r = Pm.shape[1]
        A, C = len(self.rows_by_a), len(self.cols_by_c)
        cross = np.zeros((A, C, r))
        for ci, cols in enumerate(self.cols_by_c):
            G = M[:, cols] @ Qm[cols, :]
            PG = Pm * G
            for ai, rows in enumerate(self.rows_by_a):
                cross[ai, ci] = PG[rows].sum(0)
        cross = np.cumsum(cross, axis=2)
        PPa = [Pm[rows].T @ Pm[rows] for rows in self.rows_by_a]
        QQc = [Qm[cols].T @ Qm[cols] for cols in self.cols_by_c]
        res = {}
        for chi in chis:
            rr = min(chi, r)
            S = np.zeros((A, C))
            for ai in range(A):
                for ci in range(C):
                    S[ai, ci] = self.M2[ai, ci] - 2 * cross[ai, ci, rr - 1] + float(np.sum(PPa[ai][:rr, :rr] * QQc[ci][:rr, :rr]))
            S = np.maximum(S, 0.0)
            res[chi] = dict(S=S.tolist(), fro2=float(S.sum()), dC=self.C_of_factors(Pm[:, :rr], Qm[:, :rr]).tolist())
        return res
