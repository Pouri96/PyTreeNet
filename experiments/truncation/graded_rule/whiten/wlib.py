"""Shared numerics for the closed-form whitened truncation first move (everything in float64).

``CutLin`` rebuilds, for one two-site matrix M (2l x 2r) and one SPCFast configuration, the same objective as rule/spcfast.py
(region state W = Lm M Rm, normalised Hermitian-basis vector h(rho), residual r = F (h(rho_k) - h(rho))) but with its own
M-space linearisation, so that dense Jacobians and per-row gradients Gamma_i (2l x 2r) are available:

    dr_i = Re <Gamma_i, dM>,   <A, B> = sum conj(A) B,   dM = M-space perturbation at the base point.

Conventions used by the Kronecker / whitening helpers
    quadratic form   q(dM) = sum_i |<Gamma_i, dM>|^2 = Tr(dM^H  L  dM  R)  (rank-1 Gamma = u v^H gives L = u u^H, R = v v^H)
    Shampoo factors  L = sum_i Gamma_i Gamma_i^H  (2l x 2l),   R = sum_i Gamma_i^H Gamma_i  (2r x 2r)
    whitened SVD     X = SVD_k(L^{1/2} M R^{1/2}),   M_k = L^{-1/2} X R^{-1/2}
"""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------- Kronecker
def vlp_rank1_real(G, n1, n2):
    """Van Loan-Pitsianis nearest Kronecker product of the real symmetric G ((n1 n2) x (n1 n2), index (i1, i2), i1 major):
    G ~ kron(A, B), A n1 x n1, B n2 x n2.  Returns A, B (symmetrised, sign fixed to positive trace) and the captured fraction
    sigma_1^2 / sum sigma^2 of the rearranged matrix."""
    R = G.reshape(n1, n2, n1, n2).transpose(0, 2, 1, 3).reshape(n1 * n1, n2 * n2)
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    A = np.sqrt(s[0]) * u[:, 0].reshape(n1, n1)
    B = np.sqrt(s[0]) * vt[0].reshape(n2, n2)
    A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
    if np.trace(A) < 0:
        A, B = -A, -B
    return A, B, float(s[0] ** 2 / np.sum(s ** 2))


def gram_G1(Gam):
    """G1[(a,b),(c,d)] = sum_i Gam_i[a,b] conj(Gam_i[c,d])  (so that q(dM) = sum_i |<Gam_i,dM>|^2 = dM^H G1 dM, row-major vec)"""
    n, na, nb = Gam.shape
    g = Gam.reshape(n, na * nb)
    return g.T @ g.conj()


def vlp_rank1_gamma(Gam, G1=None):
    """Nearest Kronecker factorisation of the Hermitian Gauss-Newton form G1.  Returns L (na x na), R (nb x nb) Hermitian with
    q(dM) ~ Tr(dM^H L dM R) and the captured fraction of the rearranged matrix."""
    n, na, nb = Gam.shape
    if G1 is None:
        G1 = gram_G1(Gam)
    Rm = G1.reshape(na, nb, na, nb).transpose(0, 2, 1, 3).reshape(na * na, nb * nb)
    u, s, vh = np.linalg.svd(Rm, full_matrices=False)
    L = np.sqrt(s[0]) * u[:, 0].reshape(na, na)
    Rv = np.sqrt(s[0]) * vh[0].reshape(nb, nb)                 # R ~ s0 u0 vh0  ->  G1[(a,b),(c,d)] ~ L[a,c] Rv[b,d]
    Rt = Rv.T                                                  # q = dM*_{ab} L_{ac} Rv_{bd} dM_{cd} = Tr(dM^H L dM Rv^T)
    L, Rt = 0.5 * (L + L.conj().T), 0.5 * (Rt + Rt.conj().T)
    ph = np.trace(L)
    if abs(ph) > 0:
        ph = ph / abs(ph)
        L, Rt = L / ph, Rt * ph
    if np.trace(L).real < 0:
        L, Rt = -L, -Rt
    return L, Rt, float(s[0] ** 2 / np.sum(s ** 2))


def shampoo(Gam):
    """L = sum Gamma Gamma^H, R = sum Gamma^H Gamma"""
    L = np.tensordot(Gam, Gam.conj(), axes=([0, 2], [0, 2]))
    R = np.tensordot(Gam.conj(), Gam, axes=([0, 1], [0, 1]))
    return L, R


def herm_pow(H, p, floor=0.0):
    """H^p for Hermitian PSD H via eigh (eigenvalues clipped at floor)"""
    w, V = np.linalg.eigh(0.5 * (H + H.conj().T))
    w = np.maximum(w, floor)
    return (V * w ** p) @ V.conj().T


def damped_pair(Lh, Rh, mu):
    """Lw = I + mu Lh / mean-eig, Rw = I + mu Rh / mean-eig  (mean eigenvalue normalisation makes mu a dimensionless ratio lambda^-1 mu)"""
    Ln = Lh / (np.trace(Lh).real / Lh.shape[0])
    Rn = Rh / (np.trace(Rh).real / Rh.shape[0])
    return np.eye(Lh.shape[0]) + mu * Ln, np.eye(Rh.shape[0]) + mu * Rn


def whitened_subspace(Mm, k, Lw, Rw, dirn):
    """closed-form weighted rank-k approximation min ||Lw^{1/2} (M - M_k) Rw^{1/2}||_F and the orthonormal basis (2l x k for dirn 'R'
    column space, 2r x k for 'L' row space) of M_k."""
    Lh, Rh = herm_pow(Lw, 0.5), herm_pow(Rw, 0.5)
    U, s, Vh = np.linalg.svd(Lh @ Mm @ Rh, full_matrices=False)
    if dirn == 'R':
        Q, _ = np.linalg.qr(herm_pow(Lw, -0.5) @ U[:, :k])
    else:
        Q, _ = np.linalg.qr(herm_pow(Rw, -0.5) @ Vh[:k].conj().T)
    return Q, s


def weighted_lra_full(Mm, k, Lw, Rw):
    """the whole weighted rank-k approximation M_k (used by the validation tests)"""
    Lh, Rh = herm_pow(Lw, 0.5), herm_pow(Rw, 0.5)
    U, s, Vh = np.linalg.svd(Lh @ Mm @ Rh, full_matrices=False)
    X = (U[:, :k] * s[:k]) @ Vh[:k]
    return herm_pow(Lw, -0.5) @ X @ herm_pow(Rw, -0.5)


def kron_solve(A, B, g, delta_rel):
    """(A kron B + delta I)^{-1} g for symmetric/Hermitian A (n1), B (n2); g has length n1 n2 (index i1 major); delta = delta_rel * lmax(A) lmax(B)"""
    la, Ua = np.linalg.eigh(0.5 * (A + A.conj().T))
    lb, Ub = np.linalg.eigh(0.5 * (B + B.conj().T))
    la, lb = np.maximum(la, 0.0), np.maximum(lb, 0.0)
    delta = delta_rel * la.max() * lb.max()
    G = g.reshape(A.shape[0], B.shape[0])
    Gh = Ua.conj().T @ G @ Ub.conj()
    Gh = Gh / (la[:, None] * lb[None, :] + delta)
    return (Ua @ Gh @ Ub.T).reshape(-1)


def realify(H):
    """real representation of a Hermitian matrix acting on [Re z; Im z]: z^H H z = x^T rho(H) x"""
    return np.block([[H.real, -H.imag], [H.imag, H.real]])


# ---------------------------------------------------------------------------------------------------------------- the cut
class CutLin:
    def __init__(self, sp, theta, chi, dirn, b, M_mod):
        self.sp, self.dirn, self.b, self.chi = sp, dirn, b, chi
        l, r = theta.shape[0], theta.shape[3]
        self.l, self.r = l, r
        Mm = theta.reshape(2 * l, 2 * r)
        self.Mm = Mm
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M_mod._rank(s, chi)
        self.U, self.s, self.Vh, self.k = U, s, Vh, k
        self.nb = len(s) - k
        self.nrm2 = float(np.sum(s ** 2))
        self.tail = float(np.sum(s[k:] ** 2) / self.nrm2)
        lo, hi = max(0, b - sp.aL), min(sp.N, b + 2 + sp.aR)
        self.lo, self.hi = lo, hi
        D, (Fs, Fd), iu, dg = sp._region(hi - lo, hi == sp.N, b - lo)
        self.D, self.iu, self.dg = D, iu, dg
        self.F = np.concatenate([Fs.toarray(), Fd], axis=0) if Fs.shape[0] else np.array(Fd)
        self.ns = Fs.shape[0]
        self.nres = self.F.shape[0]
        self.nd, self.nu = len(dg), len(iu[0])
        self.nh = D * D
        Lm, Rm = sp._maps(lo, hi, b, l, r)
        self.Lm, self.Rm = Lm, Rm
        self.no, self.nX = Lm.shape[0], Lm.shape[1]
        self.nZ, self.npp = Rm.shape[1], Rm.shape[2]
        self.LmM, self.RmM = Lm.reshape(self.no * self.nX, l), Rm.reshape(r, self.nZ * self.npp)
        if dirn == 'R':
            self.Bk, self.Bp = U[:, :k], U[:, k:]
        else:
            self.Bk, self.Bp = Vh[:k].conj().T, Vh[k:].conj().T
        self.Mk0 = (U[:, :k] * s[:k]) @ Vh[:k]
        self.ones_d = np.zeros(self.nh)
        self.ones_d[:self.nd] = 1.0
        self.ht = self.hnorm(Mm)[0]
        hn0, self.tr0 = self.hnorm(self.Mk0)
        self.hn0 = hn0
        self.r0 = self.F @ (hn0 - self.ht)
        self.f_svd = float(self.r0 @ self.r0)
        self.W0 = self.Wof(self.Mk0)
        if dirn == 'R':
            Ak, Ap = self.Bk.conj().T @ Mm, self.Bp.conj().T @ Mm
            self.P1, self.Q1, self.P2, self.Q2 = self.Bp, Ak, self.Bk, Ap
        else:
            self.P1, self.Q1, self.P2, self.Q2 = Mm @ self.Bp, self.Bk.conj().T, Mm @ self.Bk, self.Bp.conj().T
        self.nc = self.nb * k
        self.nx = 2 * self.nc

    # ---- forward map
    def Wof(self, M2):
        l, r = self.l, self.r
        Y = self.LmM @ M2.reshape(l, 4 * r)
        Y = Y.reshape(self.no * self.nX * 4, r) @ self.RmM
        return Y.reshape(self.no, self.nX, 4, self.nZ, self.npp).transpose(1, 2, 3, 0, 4).reshape(self.D, self.no * self.npp)

    def Wadj(self, G):
        """complex adjoint of Wof; G (..., D, no npp) -> (..., 2l, 2r)"""
        lead = G.shape[:-2]
        G5 = G.reshape(-1, self.nX, 4, self.nZ, self.no, self.npp)
        out = np.empty((G5.shape[0], 2 * self.l, 2 * self.r), dtype=complex)
        for i in range(G5.shape[0]):
            Y = G5[i].transpose(3, 0, 1, 2, 4).reshape(self.no * self.nX * 4, self.nZ * self.npp) @ self.RmM.conj().T
            Y = Y.reshape(self.no * self.nX, 4 * self.r)
            out[i] = (self.LmM.conj().T @ Y).reshape(2 * self.l, 2 * self.r)
        return out.reshape(lead + (2 * self.l, 2 * self.r))

    def hvec(self, rho):
        return np.concatenate([rho[self.dg, self.dg].real, rho[self.iu].real, rho[self.iu].imag])

    def hnorm(self, M2):
        W = self.Wof(M2)
        rho = W @ W.conj().T
        tr = rho[self.dg, self.dg].real.sum()
        return self.hvec(rho) / tr, tr

    def resid(self, M2):
        return self.F @ (self.hnorm(M2)[0] - self.ht)

    def fval(self, M2):
        r = self.resid(M2)
        return float(r @ r), r

    # ---- linearisation at the base point Mk0
    def jvp_M(self, dM):
        dW = self.Wof(dM)
        X1 = dW @ self.W0.conj().T
        drho = X1 + X1.conj().T
        dh = self.hvec(drho)
        dh = (dh - self.hn0 * dh[:self.nd].sum()) / self.tr0
        return self.F @ dh

    def gamma_rows(self):
        """Gamma_i (nres, 2l, 2r) with dr_i = Re <Gamma_i, dM>"""
        Eta = self.F - np.outer(self.F @ self.hn0, self.ones_d)
        Eta = Eta / self.tr0
        D, nd, nu = self.D, self.nd, self.nu
        H = np.zeros((self.nres, D, D), dtype=complex)
        H[:, self.iu[0], self.iu[1]] = (Eta[:, nd:nd + nu] + 1j * Eta[:, nd + nu:]) / 2
        H = H + H.conj().transpose(0, 2, 1)
        H[:, self.dg, self.dg] = Eta[:, :nd]
        Gd = 2.0 * np.matmul(H, self.W0[None])
        return self.Wadj(Gd)

    def jac_M(self, Gam):
        n = Gam.shape[0]
        g = Gam.reshape(n, -1)
        return np.concatenate([g.real, g.imag], axis=1)

    # ---- tangent chart (as rule/spcfast.py make_chart, two=False, free=0)
    def dM_of_x(self, x):
        nc, nb, k = self.nc, self.nb, self.k
        C = (x[:nc] + 1j * x[nc:]).reshape(nb, k)
        return self.P1 @ C @ self.Q1 + self.P2 @ C.conj().T @ self.Q2

    def T_dense(self):
        n2 = 4 * self.l * self.r
        T = np.empty((2 * n2, self.nx))
        e = np.zeros(self.nx)
        for j in range(self.nx):
            e[j] = 1.0
            d = self.dM_of_x(e).ravel()
            T[:n2, j], T[n2:, j] = d.real, d.imag
            e[j] = 0.0
        return T

    def retract(self, x, alpha=1.0):
        nc, nb, k = self.nc, self.nb, self.k
        z = alpha * (x[:nc] + 1j * x[nc:])
        C = z.reshape(nb, k)
        Q, _ = np.linalg.qr(self.Bk + self.Bp @ C)
        return Q, self.galerkin(Q)

    def galerkin(self, Q):
        return Q @ (Q.conj().T @ self.Mm) if self.dirn == 'R' else (self.Mm @ Q) @ Q.conj().T

    def C_of_Q(self, Q):
        """tangent coordinates (real vector) of the subspace Q, or None if Q is (nearly) orthogonal to the SVD subspace"""
        A = self.Bk.conj().T @ Q
        if np.linalg.cond(A) > 1e10:
            return None
        C = (self.Bp.conj().T @ Q) @ np.linalg.inv(A)
        z = C.ravel()
        return np.concatenate([z.real, z.imag])

    def span2(self, r):
        return self.sp._low_span_resid(r, self.hi - self.lo)

    def disc_ratio(self, Mk):
        return (self.nrm2 - float(np.vdot(Mk, Mk).real)) / max(self.nrm2 * self.tail, 1e-300)

    # ---- diag preconditioner of rule/spcfast.py
    def minv_diag(self):
        cs = self.s[:self.k] ** 2
        sc = np.tile(np.broadcast_to(cs[None, :], (self.nb, self.k)).ravel(), 2) / self.nrm2
        return 1.0 / (3.0 * sc + 1e-9 * float(sc.max()))
