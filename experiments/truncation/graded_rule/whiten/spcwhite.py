"""SPCWhite(SPCFast): closed-form whitened-SVD cut (oracle-grade Shampoo factors), for the end-to-end check T3-lite.

Same firing gates, region objective and return format as rule/spcfast.SPCFast; only the step differs.  At a fired cut

    Gamma_i = d r_i / d M   (dense, float64, via wlib.CutLin: one adjoint per residual row, so this is NOT a cheap implementation)
    L = sum Gamma_i Gamma_i^H,  R = sum Gamma_i^H Gamma_i,   normalised to unit mean eigenvalue
    Lw = I + mu L, Rw = I + mu R,   X = SVD_k(Lw^{1/2} M Rw^{1/2}),   Q = orth(col(Lw^{-1/2} U_k))  (row space for dirn 'L')
    kept tensors:  Q and the Galerkin centre Q^H M  (or M Q), as everywhere else in the code base.

modes
  'all'     all residual rows (static + time-evolved), linearised at the SVD point            (probe arm A5)
  'static'  static (tau = 0) rows only, linearised at the untruncated M: nothing but M and the  (probe arm A5sm)
            transferred static strings enters, i.e. the target-free ingredients of the planned 'core' mode
accept=True additionally evaluates the exact region objective once and falls back to the SVD cut if it did not drop (the "3x" variant).

NOT implemented: the cheap 'kfac' (Kronecker GN step; probe arm A4 failed, median retention ~0.1) and the cheap 'core' construction of L, R
without region objects (T2, gated on T1 which came out ambiguous).  Timings of this class say nothing about a cheap implementation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
import wlib as W


class SPCWhite(spcfast.SPCFast):
    def __init__(self, *args, mode='all', mu=1e4, accept=False, **kw):
        super().__init__(*args, **kw)
        self.mode, self.mu, self.accept = mode, mu, accept
        self.rejected = 0
        self.wlog = []

    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        nrm2 = float(np.sum(s ** 2))
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.tmax = max(self.tmax, tail)
        if tail < self.rel_skip * self.tmax:
            self.skipped += 1
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        if b < self.bmin or b > self.bmax:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        if (self.pattern in ('R', 'L') and dirn != self.pattern) or (self.pattern == 'alt' and (b % 2 == 0) != (dirn == 'R')):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        cl = W.CutLin(self, theta, chi, dirn, b, M)
        if self.mode == 'all':
            Gam = cl.gamma_rows(at='svd', rows='all')
        elif self.mode == 'static':
            Gam = cl.gamma_rows(at='full', rows='static')
        else:
            raise ValueError(self.mode)
        Ls, Rs = W.shampoo(Gam)
        Lw, Rw = W.damped_pair(Ls, Rs, self.mu)
        Q, _ = W.whitened_subspace(cl.Mm, k, Lw, Rw, dirn)
        if self.accept:
            f, _ = cl.fval(cl.galerkin(Q))
            if not f < cl.f_svd * (1 - 1e-3):
                self.rejected += 1
                return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        kept = float(np.linalg.norm(cl.galerkin(Q)) ** 2)
        self.log.append((cl.f_svd, cl.fval(cl.galerkin(Q))[0] if self.accept else float('nan'), (nrm2 - float(np.sum(s[:k] ** 2))) / nrm2, (nrm2 - kept) / nrm2, b,
                         1.0 if dirn == 'R' else 0.0, tail))
        if dirn == 'R':
            cen = Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, k), Q.conj().T.reshape(k, 2, r), 0.0, True
