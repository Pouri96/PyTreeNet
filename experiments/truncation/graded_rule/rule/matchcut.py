"""Marginal-matching cut: a rank-k two-site tensor whose window reduced density matrix matches the untruncated one.

At a cut the two-site tensor theta (l,2,2,r) has a left-isometric neighbour A and a right-isometric
neighbour B, so the window marginal is rho(M) = W W^dag, W = A M B (mpsenh.make_wide). Plain SVD keeps the
fidelity-optimal rank-k tensor M_0 and leaves rho(M_0) != tau := rho(theta). This cut instead solves
    min_M  || S^{1/2} (rho(M) - tau) ||_F   over rank-k M
by Gauss-Newton on the fixed-rank manifold. Each step is the minimum-norm tangent correction that removes
the linearised residual, found from a (D^2 x D^2) Gram system (D = window dimension), followed by a rank-k
SVD retraction. A trust region on the fidelity cost 1 - |<theta, M>|^2 <= (1 + kappa) eps is optional
(kappa = inf means none). kappa = 0 returns the SVD cut.
"""
import numpy as np
import mpsenh as M


def _hermitian_basis(D):
    """Real-orthonormal basis of the D x D Hermitian matrices, shape (D^2, D, D)."""
    E = []
    for i in range(D):
        m = np.zeros((D, D), dtype=complex)
        m[i, i] = 1.0
        E.append(m)
    for i in range(D):
        for j in range(i + 1, D):
            m = np.zeros((D, D), dtype=complex)
            m[i, j] = m[j, i] = 1.0 / np.sqrt(2)
            E.append(m)
            m = np.zeros((D, D), dtype=complex)
            m[i, j] = -1j / np.sqrt(2)
            m[j, i] = 1j / np.sqrt(2)
            E.append(m)
    return np.array(E)


class MatchCut:
    def __init__(self, model, N, window='r2', gamma=1.0, kappa=np.inf, iters=4, mu=1e-10, tol=1e-10):
        self.model, self.N = model, N
        self.window, self.gamma, self.kappa, self.iters, self.mu, self.tol = window, gamma, kappa, iters, mu, tol
        self.wins = {}
        self.fired = 0
        self.truncating = 0
        self.stats = []          # per cut: (f_svd, f_final, cost_ratio, iterations)

    # ------------------------------------------------------------------ window data, cached per bond
    def _win(self, b, hasA, hasB):
        key = (b, hasA, hasB)
        if key not in self.wins:
            w = M.Window(self.model, self.N, b, hasA, hasB, self.gamma)
            D = w.D
            E = _hermitian_basis(D)
            S = w.S_obj
            ev, evec = np.linalg.eigh(0.5 * (S + S.conj().T))
            sqrtS = (evec * np.sqrt(np.clip(ev, 0, None))) @ evec.conj().T
            w.E = E
            w.sqrtS = sqrtS
            w.SE = np.einsum('xy,ayz->axz', np.eye(1), E.reshape(len(E), D * D, 1)[:, :, :1].reshape(len(E), D * D, 1)) if False else \
                (E.reshape(len(E), D * D) @ sqrtS.T).reshape(len(E), D, D)
            self.wins[key] = w
        return self.wins[key]

    # ------------------------------------------------------------------
    def __call__(self, theta, chi, dirn, A, B, b):
        l, r = theta.shape[0], theta.shape[3]
        Mth = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(Mth, full_matrices=False)
        k = M._rank(s, chi)
        if k >= len(s) or np.sum(s[k:] ** 2) <= 0 or self.kappa <= 0:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.truncating += 1
        eps = float(np.sum(s[k:] ** 2) / np.sum(s ** 2))
        out = self._match(theta, U, s, Vh, k, eps, l, r, dirn, A, B, b)
        if out is None:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        return out

    def _match(self, theta, U, s, Vh, k, eps, l, r, dirn, A, B, b):
        if self.window == 'bond':
            win = self._win(b, False, False)

            def wide(Th):
                return Th.transpose(1, 2, 0, 3).reshape(4, -1)

            def back(Ym):
                return Ym.reshape(2, 2, l, r).transpose(2, 0, 1, 3)
        else:
            hasA, hasB = A is not None, B is not None
            win = self._win(b, hasA, hasB)
            Ae = A if hasA else np.eye(l, dtype=complex).reshape(l, 1, l)
            Be = B if hasB else np.eye(r, dtype=complex).reshape(r, 1, r)
            wide, back = M.make_wide(Ae, Be, win)
        D, D2 = win.D, win.D * win.D
        nth = np.linalg.norm(theta)
        th = theta / nth
        Wt = wide(th)
        tau = Wt @ Wt.conj().T

        def f_of(rho):
            d = (rho - tau).reshape(-1)
            return win.cW * float(np.vdot(d, win.S_obj @ d).real)

        def cost_of(Mm):
            return 1.0 - abs(np.vdot(th.reshape(l * 2, 2 * r), Mm)) ** 2 / float(np.vdot(Mm, Mm).real)

        def rank_k(Mm):
            Uu, ss, Vv = np.linalg.svd(Mm, full_matrices=False)
            Mk = (Uu[:, :k] * ss[:k]) @ Vv[:k]
            return Mk / np.linalg.norm(Mk), Uu[:, :k], Vv[:k]

        Mc, Uk, Vk = rank_k((U[:, :k] * s[:k]) @ Vh[:k] / np.linalg.norm(s))
        Vk = Vk.conj().T
        rho = None
        Wc = wide(Mc.reshape(l, 2, 2, r))
        rho = Wc @ Wc.conj().T
        f0 = f_of(rho)
        f = f0
        used = 0
        for it in range(self.iters):
            if f < self.tol * max(f0, 1e-300) or f < 1e-28:
                break
            res = rho - tau
            sres = (win.sqrtS @ res.reshape(-1)).reshape(D, D)
            c = np.array([np.vdot(Ea, sres).real for Ea in win.E])
            Phi = np.empty((D2, 4 * l * r), dtype=complex)
            for a in range(D2):
                Z = back(2.0 * win.SE[a] @ Wc).reshape(l * 2, 2 * r)
                UZ = Uk.conj().T @ Z
                Z = Uk @ UZ + (Z @ Vk) @ Vk.conj().T - Uk @ ((UZ @ Vk) @ Vk.conj().T)
                Phi[a] = Z.reshape(-1)
            K = (Phi.conj() @ Phi.T).real
            lam = np.linalg.solve(K + self.mu * np.trace(K) / D2 * np.eye(D2), -c)
            delta = (lam @ Phi).reshape(l * 2, 2 * r)
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
                Mn, Un, Vn = rank_k(Mc + alpha * delta)
                Wn = wide(Mn.reshape(l, 2, 2, r))
                rn = Wn @ Wn.conj().T
                fn = f_of(rn)
                if fn < f and cost_of(Mn) <= (1.0 + self.kappa) * eps:
                    accepted = True
                    break
            if not accepted:
                break
            Mc, Uk, Vk, rho, Wc, f = Mn, Un, Vn.conj().T, rn, Wn, fn
            used += 1
        if used == 0:
            return None
        self.stats.append((f0, f, cost_of(Mc) / eps, used))
        Uu, ss, Vv = np.linalg.svd(Mc, full_matrices=False)
        ss = ss[:k]
        Uu, Vv = Uu[:, :k], Vv[:k]
        nrm = np.linalg.norm(ss)
        if dirn == 'R':
            return Uu.reshape(l, 2, k), ((ss[:, None] / nrm) * Vv).reshape(k, 2, r), 0.0, True
        return (Uu * (ss / nrm)).reshape(l, 2, k), Vv.reshape(k, 2, r), 0.0, True
